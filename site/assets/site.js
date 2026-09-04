(function () {
  "use strict";

  const body = document.body;
  const page = body.dataset.page;
  const root = body.dataset.siteRoot || ".";

  function text(element, value) {
    element.textContent = value;
    return element;
  }

  function formatMetric(value, name) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    if (name === "mean_realized_violations_at_p01_b02") return number.toFixed(3);
    if (name.includes("violations")) return Math.round(number).toLocaleString();
    if (Math.abs(number) >= 100) return number.toFixed(1);
    return number.toFixed(3);
  }

  function labelMetric(name) {
    if (name === "mean_realized_violations_at_p01_b02") {
      return "Mean violations (1% attacks, 2% audit)";
    }
    return name
      .replaceAll("_at_0.02", " @ 2%")
      .replaceAll("_at_0.05", " @ 5%")
      .replaceAll("_at_0.1", " @ 10%")
      .replaceAll("_", " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function isReference(row) {
    const status = String(row.result_status || "").toLowerCase();
    return status.includes("oracle") || status.includes("evaluator-only bound") || status.includes("reference bound") || status.includes("exact no-op");
  }

  function statusClass(row) {
    if (isReference(row)) return "reference";
    const status = String(row.result_status || "").toLowerCase();
    if (status.includes("post-outcome") || status.includes("bounded") || status.includes("open replay")) return "secondary";
    return "checked";
  }

  function isEligible(row) {
    if (isReference(row)) return false;
    const status = String(row.result_status || "").toLowerCase();
    return status.includes("prespecified") || status.includes("prospectively frozen") || status.includes("outside submission") || status.includes("community evaluated");
  }

  function isSecondary(row) {
    return !isReference(row) && !isEligible(row);
  }

  async function loadJson(url) {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) throw new Error(`Could not load ${url} (${response.status})`);
    return response.json();
  }

  async function setupLeaderboards() {
    const taskSelect = document.getElementById("task-select");
    const metricSelect = document.getElementById("metric-select");
    const statusSelect = document.getElementById("status-select");
    const tableBody = document.getElementById("leaderboard-body");
    const summary = document.getElementById("leaderboard-summary");
    const metricHeading = document.getElementById("metric-heading");
    const errorBox = document.getElementById("leaderboard-error");

    try {
      const catalog = await loadJson(`${root}/data/catalog.json`);
      const panels = catalog.leaderboards || [];
      if (!panels.length) throw new Error("No checked leaderboard panels were found.");
      taskSelect.replaceChildren();
      panels.forEach((panel, index) => {
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = `${panel.title} (${panel.row_count} rows)`;
        taskSelect.append(option);
      });

      let activeRows = [];
      let activePanel = null;

      async function loadPanel() {
        activePanel = panels[Number(taskSelect.value)];
        const payload = await loadJson(`${root}/${activePanel.data_url}`);
        activeRows = payload.rows || [];
        metricSelect.replaceChildren();
        activePanel.available_metrics.forEach((metric) => {
          const option = document.createElement("option");
          option.value = metric;
          option.textContent = labelMetric(metric);
          option.selected = metric === activePanel.primary_metric;
          metricSelect.append(option);
        });
        render();
      }

      function render() {
        const metric = metricSelect.value;
        const direction = activePanel.metric_directions[metric] || "lower";
        const filter = statusSelect.value;
        let rows = activeRows.filter((row) => {
          if (filter === "reference") return isReference(row);
          if (filter === "eligible") return isEligible(row);
          if (filter === "secondary") return isSecondary(row);
          return true;
        });
        rows = rows.slice().sort((a, b) => {
          const left = Number((a.metrics || {})[metric]);
          const right = Number((b.metrics || {})[metric]);
          return direction === "higher" ? right - left : left - right;
        });

        metricHeading.textContent = `${labelMetric(metric)} (${direction} is better)`;
        summary.replaceChildren();
        [
          ["Task", activePanel.task_id],
          ["Primary outcome", labelMetric(activePanel.primary_metric)],
          ["Panel", activePanel.status],
          ["Global rank", "never issued"],
        ].forEach(([name, value]) => {
          const item = document.createElement("span");
          const strong = text(document.createElement("strong"), `${name}: `);
          item.append(strong, document.createTextNode(value));
          summary.append(item);
        });

        tableBody.replaceChildren();
        let eligibleOrder = 0;
        rows.forEach((row) => {
          const reference = isReference(row);
          if (!reference) eligibleOrder += 1;
          const tr = document.createElement("tr");

          const order = text(document.createElement("td"), reference ? "—" : String(eligibleOrder));

          const observer = document.createElement("td");
          observer.className = "observer-cell";
          const name = text(document.createElement("strong"), row.display_name || row.observer_name || "Unnamed observer");
          const family = text(document.createElement("small"), row.observer_family || row.fit_procedure || "");
          observer.append(name, family);

          const status = document.createElement("td");
          const badge = text(document.createElement("span"), row.result_status || "unspecified");
          badge.className = `badge ${statusClass(row)}`;
          status.append(badge);

          const access = text(document.createElement("td"), row.access_regime || "unspecified");
          access.className = "access-cell";
          const score = text(document.createElement("td"), formatMetric((row.metrics || {})[metric], metric));
          tr.append(order, observer, status, access, score);
          tableBody.append(tr);
        });
        if (!rows.length) {
          const tr = document.createElement("tr");
          const td = text(document.createElement("td"), "No rows match this filter.");
          td.colSpan = 5;
          tr.append(td);
          tableBody.append(tr);
        }
      }

      taskSelect.addEventListener("change", () => loadPanel().catch(showError));
      metricSelect.addEventListener("change", render);
      statusSelect.addEventListener("change", render);
      await loadPanel();

      function showError(error) {
        errorBox.hidden = false;
        errorBox.textContent = error.message;
      }
    } catch (error) {
      errorBox.hidden = false;
      errorBox.textContent = `${error.message} Build the site with scripts/build_site.py and serve build/pages through a web server.`;
      tableBody.replaceChildren();
    }
  }

  if (page === "leaderboards") setupLeaderboards();
})();
