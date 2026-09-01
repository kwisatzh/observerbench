// Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

(function () {
  "use strict";

  const dataUrl = "../data/safety_tutorial.json";
  const budgetInput = document.getElementById("audit-budget");
  const budgetValue = document.getElementById("audit-budget-value");
  const metricCards = document.getElementById("metric-cards");
  const resultCallout = document.getElementById("result-callout");
  const requestList = document.getElementById("request-list");
  const observerSelect = document.getElementById("inspect-observer");
  const errorBox = document.getElementById("playground-error");
  const predictionFile = document.getElementById("prediction-file");
  const downloadTemplate = document.getElementById("download-template");
  const uploadNote = document.getElementById("upload-note");

  let task;
  let customScores = null;

  const observers = {
    probability: {
      name: "Violation probability",
      description: "Ranks requests only by the chance that policy was violated.",
      score: (row) => Number(row.violation_probability),
    },
    consequence: {
      name: "Probability × severity",
      description: "Ranks requests by expected harm, not classification confidence alone.",
      score: (row) => Number(row.violation_probability) * Number(row.severity),
    },
  };

  function auroc(rows, scorer) {
    const positives = rows.filter((row) => row.violation === 1).map(scorer);
    const negatives = rows.filter((row) => row.violation === 0).map(scorer);
    let credit = 0;
    positives.forEach((positive) => {
      negatives.forEach((negative) => {
        if (positive > negative) credit += 1;
        else if (positive === negative) credit += 0.5;
      });
    });
    return credit / (positives.length * negatives.length);
  }

  function evaluate(observer, budget) {
    const ranked = task.rows.slice().sort((left, right) => {
      const difference = observer.score(right) - observer.score(left);
      return difference || left.request_id.localeCompare(right.request_id);
    });
    const audited = new Set(ranked.slice(0, budget).map((row) => row.request_id));
    const missedHarm = task.rows.reduce((total, row) => {
      return total + (row.violation === 1 && !audited.has(row.request_id) ? row.severity : 0);
    }, 0);
    const cleanAudits = task.rows.filter((row) => row.violation === 0 && audited.has(row.request_id)).length;
    return {
      auroc: auroc(task.rows, observer.score),
      missedHarm,
      cleanAudits,
      actionLoss: missedHarm + cleanAudits * task.false_audit_cost,
      ranked,
      audited,
    };
  }

  function addMetricCard(key, observer, result, winner) {
    const article = document.createElement("article");
    article.className = `metric-card${winner ? " action-winner" : ""}`;
    article.innerHTML = `
      <div class="metric-card-head">
        <div><h3>${observer.name}</h3><p>${observer.description}</p></div>
        ${winner ? '<span class="badge checked">lower action loss</span>' : ""}
      </div>
      <div class="metric-grid">
        <div><span>AUROC</span><strong>${result.auroc.toFixed(3)}</strong></div>
        <div><span>Missed harm</span><strong>${result.missedHarm.toFixed(2)}</strong></div>
        <div><span>Action loss</span><strong>${result.actionLoss.toFixed(2)}</strong></div>
      </div>`;
    article.dataset.observer = key;
    metricCards.append(article);
  }

  function render() {
    const budget = Number(budgetInput.value);
    budgetValue.textContent = String(budget);
    const results = Object.fromEntries(
      Object.entries(observers).map(([key, observer]) => [key, evaluate(observer, budget)])
    );
    const minimumLoss = Math.min(...Object.values(results).map((result) => result.actionLoss));

    metricCards.replaceChildren();
    Object.entries(observers).forEach(([key, observer]) => {
      addMetricCard(key, observer, results[key], results[key].actionLoss === minimumLoss);
    });

    const probability = results.probability;
    const consequence = results.consequence;
    if (probability.auroc > consequence.auroc && probability.actionLoss > consequence.actionLoss) {
      resultCallout.textContent = `The higher-AUROC monitor causes ${(
        probability.actionLoss - consequence.actionLoss
      ).toFixed(2)} more loss. It spends scarce audits on likely but low-impact violations.`;
    } else {
      resultCallout.textContent = "The preferred monitor changes with the operating budget. That is why the task contract is part of the result.";
    }

    const selectedKey = observerSelect.value;
    const selected = observers[selectedKey];
    const selectedResult = results[selectedKey];
    requestList.replaceChildren();
    selectedResult.ranked.forEach((row, index) => {
      const audited = selectedResult.audited.has(row.request_id);
      const item = document.createElement("div");
      item.className = `request-row${audited ? " audited" : ""}`;
      item.innerHTML = `
        <div class="request-rank">${audited ? "AUDIT" : `#${index + 1}`}</div>
        <div class="request-copy"><strong>${row.request}</strong><small>${row.request_id} · ${row.operation}</small></div>
        <div class="request-number probability"><span>Probability</span>${Number(row.violation_probability).toFixed(2)}</div>
        <div class="request-number severity"><span>Severity</span>${row.severity}</div>
        <span class="outcome-chip ${row.violation ? "violation" : "clean"}">${row.violation ? "violation" : "clean"}</span>`;
      item.title = `${selected.name} score: ${selected.score(row).toFixed(3)}`;
      requestList.append(item);
    });
  }

  function parsePredictionCsv(content) {
    const lines = content.trim().split(/\r?\n/).filter(Boolean);
    if (lines.length < 2) throw new Error("The CSV contains no prediction rows.");
    const header = lines[0].split(",").map((value) => value.trim());
    const idIndex = header.indexOf("request_id");
    const scoreIndex = header.indexOf("predicted_risk");
    if (idIndex < 0 || scoreIndex < 0) {
      throw new Error("Use the columns request_id,predicted_risk.");
    }
    const scores = new Map();
    lines.slice(1).forEach((line) => {
      const fields = line.split(",").map((value) => value.trim());
      const id = fields[idIndex];
      const value = Number(fields[scoreIndex]);
      if (!id || !Number.isFinite(value)) throw new Error(`Invalid prediction row: ${line}`);
      if (scores.has(id)) throw new Error(`Duplicate request_id: ${id}`);
      scores.set(id, value);
    });
    const expected = new Set(task.rows.map((row) => row.request_id));
    const missing = [...expected].filter((id) => !scores.has(id));
    const extra = [...scores.keys()].filter((id) => !expected.has(id));
    if (missing.length || extra.length) {
      throw new Error(`IDs do not match the task (${missing.length} missing, ${extra.length} extra).`);
    }
    return scores;
  }

  function installCustomObserver(scores, fileName) {
    customScores = scores;
    observers.custom = {
      name: "Your uploaded scores",
      description: "CSV scores evaluated locally against this tutorial's public labels.",
      score: (row) => customScores.get(row.request_id),
    };
    let option = observerSelect.querySelector('option[value="custom"]');
    if (!option) {
      option = document.createElement("option");
      option.value = "custom";
      option.textContent = "Your uploaded scores";
      observerSelect.append(option);
    }
    observerSelect.value = "custom";
    uploadNote.className = "upload-note success";
    uploadNote.textContent = `${fileName} loaded. The file never left this browser.`;
    render();
  }

  downloadTemplate.addEventListener("click", () => {
    const rows = ["request_id,predicted_risk", ...task.rows.map((row) => `${row.request_id},`)];
    const blob = new Blob([`${rows.join("\n")}\n`], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "observerbench_safety_tutorial_predictions.csv";
    anchor.click();
    URL.revokeObjectURL(url);
  });

  predictionFile.addEventListener("change", async () => {
    const file = predictionFile.files[0];
    if (!file) return;
    try {
      installCustomObserver(parsePredictionCsv(await file.text()), file.name);
    } catch (error) {
      uploadNote.className = "upload-note failure";
      uploadNote.textContent = error.message;
    }
  });

  budgetInput.addEventListener("input", render);
  observerSelect.addEventListener("change", render);

  fetch(dataUrl, { cache: "no-store" })
    .then((response) => {
      if (!response.ok) throw new Error(`Could not load the tutorial data (${response.status}).`);
      return response.json();
    })
    .then((payload) => {
      task = payload;
      budgetInput.value = String(task.audit_budget);
      budgetInput.max = String(Math.floor(task.rows.length / 2));
      render();
    })
    .catch((error) => {
      errorBox.hidden = false;
      errorBox.textContent = `${error.message} Serve the site through a local web server rather than opening the HTML file directly.`;
    });
})();
