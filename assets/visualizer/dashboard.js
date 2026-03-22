(function () {
  const data = window.CPP_RL_LAB_DATA || {};
  const actionOrder = ["up", "down", "left", "right"];
  const state = {
    traceIndex: 0,
    stepIndex: 0,
    timer: null,
    speedMs: 700,
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function create(tag, className, text) {
    const element = document.createElement(tag);
    if (className) {
      element.className = className;
    }
    if (text !== undefined) {
      element.textContent = text;
    }
    return element;
  }

  function formatNumber(value, digits = 3) {
    if (typeof value !== "number" || Number.isNaN(value)) {
      return "n/a";
    }
    return value.toFixed(digits);
  }

  function width() {
    return data.layout?.width || 0;
  }

  function height() {
    return data.layout?.height || 0;
  }

  function traceList() {
    return Array.isArray(data.traces) ? data.traces : [];
  }

  function metrics() {
    return Array.isArray(data.metrics) ? data.metrics : [];
  }

  function evaluationPoints() {
    return Array.isArray(data.evaluationPoints) ? data.evaluationPoints : [];
  }

  function getTrace() {
    return traceList()[state.traceIndex] || null;
  }

  function stateToPosition(index) {
    const cols = width();
    return {
      row: Math.floor(index / cols),
      col: index % cols,
    };
  }

  function cellSymbol(row, col) {
    return data.layout.rows[row][col];
  }

  function stateAtCurrentFrame(trace, stepIndex) {
    if (!trace) {
      return 0;
    }
    if (stepIndex <= 0 || trace.stepsTrace.length === 0) {
      return trace.initialState;
    }
    const safeIndex = Math.min(stepIndex, trace.stepsTrace.length) - 1;
    return trace.stepsTrace[safeIndex].nextState;
  }

  function currentStep(trace, stepIndex) {
    if (!trace || stepIndex <= 0 || trace.stepsTrace.length === 0) {
      return null;
    }
    return trace.stepsTrace[Math.min(stepIndex, trace.stepsTrace.length) - 1];
  }

  function visitedStates(trace, stepIndex) {
    const visited = new Set();
    if (!trace) {
      return visited;
    }

    visited.add(trace.initialState);
    for (let index = 0; index < Math.min(stepIndex, trace.stepsTrace.length); index += 1) {
      visited.add(trace.stepsTrace[index].nextState);
    }
    return visited;
  }

  function globalValueRange() {
    const values = [];
    (data.stateActionValues || []).forEach((row) => {
      if (Array.isArray(row) && row.length > 0) {
        values.push(Math.max(...row));
      }
    });

    if (values.length === 0) {
      return null;
    }

    return {
      min: Math.min(...values),
      max: Math.max(...values),
    };
  }

  function heatColor(value, range) {
    if (!range || typeof value !== "number" || Number.isNaN(value)) {
      return "";
    }

    const span = Math.max(1e-9, range.max - range.min);
    const normalized = Math.max(0, Math.min(1, (value - range.min) / span));
    const hue = 24 + normalized * 140;
    const alpha = 0.12 + normalized * 0.5;
    return `hsla(${hue}, 72%, 58%, ${alpha})`;
  }

  function metricSeries(source, key) {
    return source.map((item, index) => ({
      x: item.episode || index + 1,
      y: item[key],
    }));
  }

  function drawLineChart(canvasId, series, options) {
    const canvas = byId(canvasId);
    if (!canvas) {
      return;
    }

    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const displayWidth = canvas.clientWidth || canvas.width;
    const displayHeight = canvas.clientHeight || canvas.height;
    canvas.width = displayWidth * dpr;
    canvas.height = displayHeight * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.clearRect(0, 0, displayWidth, displayHeight);
    ctx.fillStyle = "#fffaf1";
    ctx.fillRect(0, 0, displayWidth, displayHeight);

    const padding = { top: 24, right: 18, bottom: 34, left: 52 };
    const plotWidth = displayWidth - padding.left - padding.right;
    const plotHeight = displayHeight - padding.top - padding.bottom;
    const flat = series.flatMap((item) => item.points);

    if (flat.length === 0) {
      ctx.fillStyle = "#5b777d";
      ctx.font = '16px "Avenir Next", "Trebuchet MS", sans-serif';
      ctx.fillText("No chart data available for this run.", 24, displayHeight / 2);
      return;
    }

    const xValues = flat.map((point) => point.x);
    const yValues = flat.map((point) => point.y);
    const minX = Math.min(...xValues);
    const maxX = Math.max(...xValues);
    let minY = Math.min(...yValues);
    let maxY = Math.max(...yValues);

    if (minY === maxY) {
      minY -= 1;
      maxY += 1;
    }

    const toX = (value) =>
      padding.left + ((value - minX) / Math.max(1e-9, maxX - minX)) * plotWidth;
    const toY = (value) =>
      padding.top + (1 - (value - minY) / (maxY - minY)) * plotHeight;

    ctx.strokeStyle = "rgba(18, 51, 59, 0.12)";
    ctx.lineWidth = 1;
    ctx.font = '12px "Avenir Next", "Trebuchet MS", sans-serif';
    ctx.fillStyle = "#5b777d";

    for (let index = 0; index <= 4; index += 1) {
      const y = padding.top + (plotHeight / 4) * index;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(displayWidth - padding.right, y);
      ctx.stroke();
      const labelValue = maxY - ((maxY - minY) / 4) * index;
      ctx.fillText(formatNumber(labelValue, 2), 8, y + 4);
    }

    ctx.strokeStyle = "rgba(18, 51, 59, 0.2)";
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, displayHeight - padding.bottom);
    ctx.lineTo(displayWidth - padding.right, displayHeight - padding.bottom);
    ctx.stroke();

    series.forEach((item) => {
      if (!item.points.length) {
        return;
      }

      ctx.strokeStyle = item.color;
      ctx.lineWidth = item.width || 2.5;
      ctx.beginPath();
      item.points.forEach((point, index) => {
        const x = toX(point.x);
        const y = toY(point.y);
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
    });

    let legendX = padding.left;
    series.forEach((item) => {
      ctx.fillStyle = item.color;
      ctx.fillRect(legendX, 8, 18, 4);
      ctx.fillStyle = "#12333b";
      ctx.fillText(item.label, legendX + 24, 14);
      legendX += ctx.measureText(item.label).width + 56;
    });

    ctx.fillStyle = "#5b777d";
    ctx.fillText(String(minX), padding.left, displayHeight - 10);
    ctx.fillText(
      String(maxX),
      displayWidth - padding.right - ctx.measureText(String(maxX)).width,
      displayHeight - 10,
    );
    if (options?.caption) {
      ctx.fillText(options.caption, padding.left, displayHeight - 10);
    }
  }

  function renderSummary() {
    const target = byId("summary-grid");
    target.innerHTML = "";
    (data.summary || []).forEach((item) => {
      const card = create("article", "metric-card");
      card.appendChild(create("span", "metric-label", item.label));
      card.appendChild(create("strong", "metric-value", item.value));
      target.appendChild(card);
    });
  }

  function renderConfiguration() {
    const target = byId("configuration-grid");
    target.innerHTML = "";
    (data.configuration || []).forEach((item) => {
      const row = create("div", "property-row");
      row.appendChild(create("div", "property-label", item.label));
      row.appendChild(create("div", "property-value", item.value));
      target.appendChild(row);
    });
  }

  function renderMeta() {
    byId("dashboard-title").textContent = data.title || "cpp-rl-lab Visual Dashboard";
    byId("dashboard-subtitle").textContent = data.subtitle || "";
    byId("mode-pill").textContent = data.mode || "run";
    byId("agent-pill").textContent = data.agentName || "agent";
    byId("maze-pill").textContent = data.mazeSource || "maze";
  }

  function renderCharts() {
    drawLineChart(
      "reward-chart",
      [
        {
          label: "episode reward",
          color: "#d76a4a",
          points: metricSeries(metrics(), "totalReward"),
        },
        {
          label: "moving average",
          color: "#0f766e",
          points: metricSeries(metrics(), "movingAverageReward"),
          width: 3,
        },
      ],
      { caption: "episodes" },
    );

    drawLineChart(
      "control-chart",
      [
        {
          label: "success rate",
          color: "#0f766e",
          points: metricSeries(metrics(), "successRate"),
        },
        {
          label: "epsilon",
          color: "#d6a63d",
          points: metricSeries(metrics(), "epsilon"),
        },
      ],
      { caption: "episodes" },
    );

    drawLineChart(
      "evaluation-chart",
      [
        {
          label: "eval success",
          color: "#0f766e",
          points: metricSeries(evaluationPoints(), "successRate"),
        },
        {
          label: "eval avg reward",
          color: "#d76a4a",
          points: metricSeries(evaluationPoints(), "averageReward"),
        },
        {
          label: "eval avg steps",
          color: "#3d5760",
          points: metricSeries(evaluationPoints(), "averageSteps"),
        },
      ],
      { caption: "evaluation checkpoints" },
    );
  }

  function renderTraceLibrary() {
    const target = byId("trace-list");
    target.innerHTML = "";

    if (traceList().length === 0) {
      target.appendChild(create("div", "empty-state", "No recorded traces were exported for this run."));
      return;
    }

    traceList().forEach((trace, index) => {
      const card = create("article", `trace-card${index === state.traceIndex ? " active" : ""}`);
      card.addEventListener("click", () => {
        stopPlayback();
        state.traceIndex = index;
        state.stepIndex = 0;
        renderEpisodeExplorer();
        renderTraceLibrary();
      });

      const top = create("div", "trace-card-top");
      top.appendChild(create("div", "trace-card-title", `Episode ${trace.episode}`));
      top.appendChild(create("span", "trace-phase", trace.phase));
      card.appendChild(top);

      const metricsGrid = create("div", "trace-metrics");
      metricsGrid.appendChild(create("div", "", `reward ${formatNumber(trace.totalReward)}`));
      metricsGrid.appendChild(create("div", "", `steps ${trace.steps}`));
      metricsGrid.appendChild(create("div", "", `solved ${trace.solved ? "yes" : "no"}`));
      metricsGrid.appendChild(create("div", "", `epsilon ${formatNumber(trace.epsilon)}`));
      card.appendChild(metricsGrid);
      target.appendChild(card);
    });
  }

  function renderMaze() {
    const trace = getTrace();
    const target = byId("maze-grid");
    const caption = byId("maze-caption");
    target.innerHTML = "";

    if (!trace) {
      caption.textContent = "No trace selected.";
      return;
    }

    const cols = width();
    target.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;

    const currentState = stateAtCurrentFrame(trace, state.stepIndex);
    const visited = visitedStates(trace, state.stepIndex);
    const range = globalValueRange();

    for (let row = 0; row < height(); row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const cell = create("div", "maze-cell");
        const symbol = cellSymbol(row, col);
        const stateIndex = row * cols + col;

        if (symbol === "#") {
          cell.classList.add("wall");
          cell.appendChild(create("span", "maze-cell-label", "wall"));
          target.appendChild(cell);
          continue;
        }

        cell.classList.add("free");
        if (visited.has(stateIndex)) {
          cell.classList.add("path");
        }
        if (symbol === "S") {
          cell.classList.add("start");
        }
        if (symbol === "G") {
          cell.classList.add("goal");
        }
        if (stateIndex === currentState) {
          cell.classList.add("current");
        }

        const values = data.stateActionValues?.[stateIndex];
        const value = Array.isArray(values) && values.length ? Math.max(...values) : null;
        const background = heatColor(value, range);
        if (background) {
          cell.style.background = `linear-gradient(180deg, ${background}, rgba(255, 249, 239, 0.92))`;
        }

        const label = symbol === "S" ? "start" : symbol === "G" ? "goal" : `${row},${col}`;
        cell.appendChild(create("span", "maze-cell-label", label));
        if (value !== null) {
          cell.appendChild(create("span", "maze-cell-value", formatNumber(value, 2)));
        }
        target.appendChild(cell);
      }
    }

    const step = currentStep(trace, state.stepIndex);
    const status = step
      ? `Action ${step.action} produced reward ${formatNumber(step.reward)}. Cumulative reward is ${formatNumber(step.cumulativeReward)}.`
      : "Initial state before the first move. Heat coloring shows the best learned Q-value available per cell.";
    caption.textContent = status;
  }

  function renderStepDetails() {
    const trace = getTrace();
    const step = currentStep(trace, state.stepIndex);
    const target = byId("step-details");
    target.innerHTML = "";

    if (!trace) {
      target.appendChild(create("div", "empty-state", "No trace loaded."));
      return;
    }

    const headline = create("div", "detail-card");
    headline.appendChild(create("div", "detail-title", "Playback State"));
    const grid = create("div", "detail-grid");
    [
      ["Episode", String(trace.episode)],
      ["Phase", trace.phase],
      ["Current Step", `${state.stepIndex} / ${trace.stepsTrace.length}`],
      ["Solved", trace.solved ? "yes" : "no"],
      ["Total Reward", formatNumber(trace.totalReward)],
      ["Epsilon", formatNumber(trace.epsilon)],
    ].forEach(([label, value]) => {
      const item = create("div", "");
      item.appendChild(create("span", "property-label", label));
      item.appendChild(create("strong", "", value));
      grid.appendChild(item);
    });
    headline.appendChild(grid);
    target.appendChild(headline);

    const decision = create("div", "detail-card");
    decision.appendChild(create("div", "detail-title", "Current Decision"));
    if (!step) {
      decision.appendChild(create("p", "panel-copy", "The explorer is at the episode start. Press play or move the slider to inspect each decision."));
    } else {
      const decisionGrid = create("div", "detail-grid");
      [
        ["Action", step.action],
        ["Reward", formatNumber(step.reward)],
        ["Cumulative", formatNumber(step.cumulativeReward)],
        ["Transition", `${step.state} -> ${step.nextState}`],
        ["Blocked", step.blocked ? "yes" : "no"],
        ["Terminal", step.done ? "yes" : "no"],
      ].forEach(([label, value]) => {
        const item = create("div", "");
        item.appendChild(create("span", "property-label", label));
        item.appendChild(create("strong", "", value));
        decisionGrid.appendChild(item);
      });
      decision.appendChild(decisionGrid);

      const badges = create("div", "badge-row");
      const validActions = step.validActions || [];
      if (validActions.length) {
        validActions.forEach((action) => {
          badges.appendChild(create("span", "badge", action));
        });
      } else {
        badges.appendChild(create("span", "badge neutral", "no valid actions exported"));
      }
      if (step.blocked) {
        badges.appendChild(create("span", "badge warn", "wall collision"));
      }
      if (step.truncated) {
        badges.appendChild(create("span", "badge warn", "max steps reached"));
      }
      if (step.solved) {
        badges.appendChild(create("span", "badge", "goal reached"));
      }
      decision.appendChild(badges);
    }
    target.appendChild(decision);
  }

  function renderActionValues() {
    const target = byId("action-values");
    target.innerHTML = "";

    const trace = getTrace();
    if (!trace) {
      return;
    }

    const step = currentStep(trace, state.stepIndex);
    const currentState = stateAtCurrentFrame(trace, state.stepIndex);
    const values =
      (step && Array.isArray(step.actionValues) && step.actionValues.length
        ? step.actionValues
        : data.stateActionValues?.[currentState]) || [];

    const card = create("div", "detail-card");
    card.appendChild(create("div", "detail-title", "Action Values"));

    if (!values.length) {
      card.appendChild(create("p", "panel-copy", "This agent does not expose state-action values, so the dashboard is showing trajectory-level details only."));
      target.appendChild(card);
      return;
    }

    const maxMagnitude = Math.max(1e-9, ...values.map((value) => Math.abs(value)));
    actionOrder.forEach((action, index) => {
      const row = create("div", `action-row${step && step.action === action ? " active" : ""}`);
      const header = create("div", "action-header");
      header.appendChild(create("strong", "", action));
      header.appendChild(create("span", "property-label", formatNumber(values[index], 4)));
      row.appendChild(header);

      const bar = create("div", "action-bar");
      const fill = create("div", "action-fill");
      fill.style.width = `${(Math.abs(values[index]) / maxMagnitude) * 100}%`;
      bar.appendChild(fill);
      row.appendChild(bar);
      card.appendChild(row);
    });

    target.appendChild(card);
  }

  function updateTraceMeta(trace) {
    const meta = byId("trace-meta");
    if (!trace) {
      meta.textContent = "";
      return;
    }
    meta.textContent = `Episode ${trace.episode} • ${trace.phase} • steps ${trace.steps} • reward ${formatNumber(trace.totalReward)}`;
  }

  function updateSlider(trace) {
    const slider = byId("step-slider");
    const max = trace ? trace.stepsTrace.length : 0;
    slider.max = String(max);
    slider.value = String(Math.min(state.stepIndex, max));
    byId("step-counter").textContent = `Step ${slider.value} / ${max}`;
  }

  function renderEpisodeExplorer() {
    const trace = getTrace();
    if (trace) {
      state.stepIndex = Math.min(state.stepIndex, trace.stepsTrace.length);
    } else {
      state.stepIndex = 0;
    }
    updateTraceMeta(trace);
    updateSlider(trace);
    renderMaze();
    renderStepDetails();
    renderActionValues();
  }

  function stopPlayback() {
    if (state.timer) {
      window.clearInterval(state.timer);
      state.timer = null;
    }
    byId("play-toggle").textContent = "Play";
  }

  function startPlayback() {
    const trace = getTrace();
    if (!trace || trace.stepsTrace.length === 0) {
      return;
    }

    stopPlayback();
    byId("play-toggle").textContent = "Pause";
    state.timer = window.setInterval(() => {
      const max = trace.stepsTrace.length;
      if (state.stepIndex >= max) {
        stopPlayback();
        return;
      }
      state.stepIndex += 1;
      renderEpisodeExplorer();
    }, state.speedMs);
  }

  function bindControls() {
    byId("play-toggle").addEventListener("click", () => {
      if (state.timer) {
        stopPlayback();
      } else {
        startPlayback();
      }
    });

    byId("restart-step").addEventListener("click", () => {
      stopPlayback();
      state.stepIndex = 0;
      renderEpisodeExplorer();
    });

    byId("prev-step").addEventListener("click", () => {
      stopPlayback();
      state.stepIndex = Math.max(0, state.stepIndex - 1);
      renderEpisodeExplorer();
    });

    byId("next-step").addEventListener("click", () => {
      stopPlayback();
      const trace = getTrace();
      const max = trace ? trace.stepsTrace.length : 0;
      state.stepIndex = Math.min(max, state.stepIndex + 1);
      renderEpisodeExplorer();
    });

    byId("step-slider").addEventListener("input", (event) => {
      stopPlayback();
      state.stepIndex = Number(event.target.value);
      renderEpisodeExplorer();
    });

    byId("speed-select").addEventListener("change", (event) => {
      state.speedMs = Number(event.target.value);
      if (state.timer) {
        startPlayback();
      }
    });

    window.addEventListener("resize", renderCharts);
  }

  function init() {
    renderMeta();
    renderSummary();
    renderConfiguration();
    renderCharts();
    renderTraceLibrary();
    bindControls();
    renderEpisodeExplorer();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
