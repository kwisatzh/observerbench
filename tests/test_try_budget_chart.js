// Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const root = path.resolve(__dirname, "..");
const html = fs.readFileSync(path.join(root, "site/try/index.html"), "utf8");
const task = JSON.parse(fs.readFileSync(path.join(root, "site/data/safety_tutorial.json"), "utf8"));
const attributes = (tag) => Object.fromEntries([...tag.matchAll(/([\w-]+)="([^"]*)"/g)].map((match) => [match[1], match[2]]));
const element = () => ({
  value: "", textContent: "", innerHTML: "", hidden: false,
  children: [], dataset: {}, attributes: {}, events: {},
  addEventListener(name, callback) { this.events[name] = callback; },
  append(child) { this.children.push(child); },
  replaceChildren() { this.children = []; },
  setAttribute(name, value) { this.attributes[name] = value; },
  querySelector() { return null; },
  focus() {}, scrollIntoView() {},
});
const elements = new Map([...html.matchAll(/<[^>]*\bid="([^"]+)"[^>]*>/g)].map((match) => [
  match[1], Object.assign(element(), attributes(match[0]), { hidden: /\shidden(?:\s|>)/.test(match[0]) }),
]));
const reviewCounts = [element(), element()];
const resizers = [];
elements.get("budget-chart-frame").clientWidth = 736;
elements.get("inspect-observer").value = "consequence";
const context = vm.createContext({
  document: {
    getElementById: (id) => elements.get(id),
    querySelectorAll: (selector) => selector === ".chart-review-count" ? reviewCounts : [],
    createElement: element,
  },
  ResizeObserver: class {
    constructor(callback) { resizers.push(callback); }
    observe() {}
  },
  fetch: async () => ({ ok: true, json: async () => task }),
});
vm.runInContext(fs.readFileSync(path.join(root, "site/try/playground.js"), "utf8"), context);

setImmediate(() => {
  assert(elements.get("decision-outcomes").hidden);
  assert(!elements.get("budget-chart").innerHTML.includes("polyline"));
  elements.get("choose-a").events.click();
  assert(!elements.get("decision-outcomes").hidden);
  assert.equal(elements.get("chart-harm-a").textContent, "24");
  assert.equal(elements.get("chart-harm-b").textContent, "6");

  for (const width of [320, 736, 1024]) {
    elements.get("budget-chart-frame").clientWidth = width;
    resizers.forEach((callback) => callback());
    assert(elements.get("budget-chart").attributes.viewBox.startsWith(`0 0 ${width} `));
    for (let budget = 1; budget <= 8; budget++) {
      elements.get("audit-budget").value = String(budget);
      elements.get("audit-budget").events.input();
      const chart = elements.get("budget-chart").innerHTML;
      assert.equal([...chart.matchAll(/<polyline /g)].length, 2);
      const marks = [...chart.matchAll(/<(circle|rect) class="chart-point [^>]+>/g)].map((match) => ({ tag: match[1], ...attributes(match[0]) }));
      assert.equal(marks.length, 16);
      assert.equal(marks.filter((mark) => mark["data-selected"] === "true").length, 2);
      for (const mark of marks) {
        const reviews = Number(mark["data-budget"]);
        const score = (row) => row.violation_probability * (mark["data-series"] === "consequence" ? row.severity : 1);
        const reviewed = new Set(task.rows.slice().sort((a, b) => score(b) - score(a) || a.request_id.localeCompare(b.request_id)).slice(0, reviews).map((row) => row.request_id));
        const harm = task.rows.reduce((total, row) => total + (row.violation && !reviewed.has(row.request_id) ? row.severity : 0), 0);
        assert.equal(Number(mark["data-harm"]), harm);
        assert.equal(mark["data-selected"], String(reviews === budget));
        const x = Number(mark.cx ?? mark.x), y = Number(mark.cy ?? mark.y);
        assert(x >= 0 && x <= width && y >= 0 && y < 340);
      }
      assert.equal(elements.get("chart-rate-a").textContent, "95%");
      assert.equal(elements.get("chart-rate-b").textContent, "70%");
      assert(reviewCounts.every((node) => node.textContent === String(budget)));
      assert.equal(elements.get("request-list").children.length, 16);
      if (budget === 6) assert.equal(elements.get("chart-harm-a").textContent, elements.get("chart-harm-b").textContent);
    }
  }
  elements.get("choose-b").events.click();
  assert.equal(elements.get("inspect-observer").value, "consequence");
  assert.equal(elements.get("chart-harm-a").textContent, "0");
  assert.equal(elements.get("chart-harm-b").textContent, "2");
  assert(elements.get("playground-error").hidden);
  console.log("PASS: chart stays inside the reveal; both curves match the scorer at all eight budgets; slider and resize updates preserve the 95%/70% warning results.");
});
