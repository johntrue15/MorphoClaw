/**
 * AutoResearchClaw — Live Knowledge Graph viewer.
 *
 * Loads docs/data/knowledge_graph.json (the cumulative snapshot) plus
 * docs/data/runs/_manifest.json (the per-run history) and renders an
 * interactive vis-network graph. Styling intentionally mirrors the
 * KnowledgeGraph.export_html helper in .github/scripts/knowledge_graph.py
 * so the look matches the per-run artifact HTML.
 *
 * Self-contained — no MkDocs-specific globals — and gracefully no-ops on
 * any page that doesn't include the #kg-app container.
 */

(function () {
  "use strict";

  const TYPE_COLORS = {
    Media: "#58a6ff",
    Specimen: "#3fb950",
    Paper: "#d29922",
    Institution: "#f85149",
    Taxon: "#bc8cff",
    MediaList: "#79c0ff",
  };
  const TYPE_SHAPES = {
    Media: "dot",
    Specimen: "diamond",
    Paper: "triangle",
    Institution: "square",
    Taxon: "hexagon",
    MediaList: "star",
  };

  // Resolve data URLs relative to the page so we work under any subpath.
  function dataUrl(name) {
    const base = new URL(".", window.location.href);
    const u = new URL(`data/${name}`, base);
    u.searchParams.set("t", Date.now().toString());
    return u.toString();
  }

  async function fetchJSON(url, fallback) {
    try {
      const res = await fetch(url, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (err) {
      console.warn(`[KG] Could not load ${url}:`, err);
      return fallback;
    }
  }

  function formatTimestamp(iso) {
    if (!iso) return "never";
    try {
      const d = new Date(iso);
      if (Number.isNaN(d.getTime())) return iso;
      return d.toLocaleString();
    } catch (_) {
      return iso;
    }
  }

  function escapeHTML(s) {
    if (s == null) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function init() {
    const root = document.getElementById("kg-app");
    if (!root) return;

    const ui = {
      graph: document.getElementById("kgGraph"),
      empty: document.getElementById("kgEmpty"),
      sidebar: document.getElementById("kgSidebar"),
      typeFilters: document.getElementById("kgTypeFilters"),
      relationFilters: document.getElementById("kgRelationFilters"),
      search: document.getElementById("kgSearch"),
      degree: document.getElementById("kgDegree"),
      degreeValue: document.getElementById("kgDegreeValue"),
      fitBtn: document.getElementById("kgFitBtn"),
      resetBtn: document.getElementById("kgResetBtn"),
      meta: document.getElementById("kgMeta"),
      runPicker: document.getElementById("kgRunPicker"),
      legend: document.getElementById("kgLegend"),
      details: document.getElementById("kgDetails"),
      detailsTitle: document.getElementById("kgDetailsTitle"),
      detailsType: document.getElementById("kgDetailsType"),
      detailsProps: document.getElementById("kgDetailsProps"),
      detailsActions: document.getElementById("kgDetailsActions"),
      detailsClose: document.getElementById("kgDetailsClose"),
    };

    if (typeof vis === "undefined" || !vis.Network) {
      ui.empty.hidden = false;
      ui.empty.innerHTML =
        "<h2>vis-network failed to load</h2>" +
        "<p>The interactive viewer requires the <code>vis-network</code> CDN script. " +
        "Check the network tab and your CSP, or download the JSON snapshot directly from " +
        "<a href=\"data/knowledge_graph.json\">data/knowledge_graph.json</a>.</p>";
      return;
    }

    const state = {
      nodes: new vis.DataSet(),
      edges: new vis.DataSet(),
      network: null,
      currentDataUrl: dataUrl("knowledge_graph.json"),
      activeTypes: new Set(),
      activeRelations: new Set(),
      minDegree: 0,
      search: "",
      // Caches of the original payload so filters can rebuild without re-fetching.
      raw: { nodes: [], edges: [], stats: {}, generated_at: null },
      nodeDegree: new Map(),
    };

    function buildGraphOptions() {
      return {
        physics: {
          solver: "forceAtlas2Based",
          forceAtlas2Based: {
            gravitationalConstant: -30,
            springLength: 80,
            springConstant: 0.04,
          },
          stabilization: { iterations: 200 },
        },
        interaction: {
          hover: true,
          tooltipDelay: 150,
          zoomView: true,
          navigationButtons: false,
        },
        edges: {
          smooth: { type: "continuous" },
          color: { color: "#8b949e", opacity: 0.45 },
          font: { size: 9, color: "#8b949e", strokeWidth: 0 },
        },
        nodes: {
          font: { size: 12, color: getCSSVar("--md-default-fg-color") || "#c9d1d9" },
          borderWidth: 1,
        },
      };
    }

    function getCSSVar(name) {
      try {
        return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
      } catch (_) {
        return "";
      }
    }

    function computeDegree(edges, idSet) {
      const deg = new Map();
      for (const e of edges) {
        if (!idSet.has(e.source) || !idSet.has(e.target)) continue;
        deg.set(e.source, (deg.get(e.source) || 0) + 1);
        deg.set(e.target, (deg.get(e.target) || 0) + 1);
      }
      return deg;
    }

    function rebuild() {
      const allTypes = new Set(state.raw.nodes.map((n) => n.type));
      const idSet = new Set(state.raw.nodes.map((n) => n.id));
      const degree = computeDegree(state.raw.edges, idSet);
      state.nodeDegree = degree;

      const q = state.search.trim().toLowerCase();

      const visibleNodeIds = new Set();
      const visNodes = [];
      for (const n of state.raw.nodes) {
        if (state.activeTypes.size && !state.activeTypes.has(n.type)) continue;
        const d = degree.get(n.id) || 0;
        if (d < state.minDegree) continue;
        if (q) {
          const hay =
            (n.label || "").toLowerCase() +
            " " +
            (n.id || "").toLowerCase() +
            " " +
            JSON.stringify(n.properties || {}).toLowerCase();
          if (!hay.includes(q)) continue;
        }
        visibleNodeIds.add(n.id);
        visNodes.push({
          id: n.id,
          label: (n.label || n.id).slice(0, 40),
          title: `${n.type}: ${n.label || n.id}`,
          group: n.type,
          color: TYPE_COLORS[n.type] || "#8b949e",
          shape: TYPE_SHAPES[n.type] || "dot",
          size: n.type === "Media" || n.type === "Taxon" ? 12 : 18,
        });
      }

      const visEdges = [];
      for (const e of state.raw.edges) {
        if (!visibleNodeIds.has(e.source) || !visibleNodeIds.has(e.target)) continue;
        if (state.activeRelations.size && !state.activeRelations.has(e.relation)) continue;
        visEdges.push({
          id: `${e.source}::${e.target}::${e.relation}`,
          from: e.source,
          to: e.target,
          label: e.relation,
          arrows: "to",
        });
      }

      state.nodes.clear();
      state.edges.clear();
      state.nodes.add(visNodes);
      state.edges.add(visEdges);

      ui.empty.hidden = visNodes.length > 0;

      // Refresh the type counts shown alongside checkboxes.
      const typeCounts = new Map();
      for (const id of visibleNodeIds) {
        const n = state.raw.nodes.find((x) => x.id === id);
        if (!n) continue;
        typeCounts.set(n.type, (typeCounts.get(n.type) || 0) + 1);
      }
      for (const t of allTypes) {
        const el = ui.typeFilters.querySelector(`[data-count="${t}"]`);
        if (el) el.textContent = typeCounts.get(t) || 0;
      }
    }

    function buildFilters() {
      const types = new Map();
      for (const n of state.raw.nodes) {
        types.set(n.type, (types.get(n.type) || 0) + 1);
      }
      const sortedTypes = [...types.keys()].sort();
      state.activeTypes = new Set(sortedTypes);
      ui.typeFilters.innerHTML = sortedTypes
        .map(
          (t) => `
            <label class="kg-checkbox">
              <input type="checkbox" data-kg-type="${escapeHTML(t)}" checked />
              <span class="kg-swatch" style="background:${TYPE_COLORS[t] || "#8b949e"}"></span>
              <span>${escapeHTML(t)}</span>
              <span class="kg-count" data-count="${escapeHTML(t)}">${types.get(t)}</span>
            </label>
          `,
        )
        .join("");

      const rels = new Map();
      for (const e of state.raw.edges) {
        rels.set(e.relation, (rels.get(e.relation) || 0) + 1);
      }
      const sortedRels = [...rels.keys()].sort();
      state.activeRelations = new Set(sortedRels);
      ui.relationFilters.innerHTML = sortedRels
        .map(
          (r) => `
            <label class="kg-checkbox">
              <input type="checkbox" data-kg-rel="${escapeHTML(r)}" checked />
              <span>${escapeHTML(r)}</span>
              <span class="kg-count">${rels.get(r)}</span>
            </label>
          `,
        )
        .join("");

      // Legend mirrors the active types.
      ui.legend.innerHTML = sortedTypes
        .map(
          (t) => `
            <div class="kg-legend-item">
              <span class="kg-swatch" style="background:${TYPE_COLORS[t] || "#8b949e"}"></span>
              <span>${escapeHTML(t)}</span>
            </div>
          `,
        )
        .join("");

      // Wire up handlers.
      ui.typeFilters.querySelectorAll("input[type=checkbox]").forEach((cb) => {
        cb.addEventListener("change", () => {
          const t = cb.getAttribute("data-kg-type");
          if (cb.checked) state.activeTypes.add(t);
          else state.activeTypes.delete(t);
          rebuild();
        });
      });
      ui.relationFilters.querySelectorAll("input[type=checkbox]").forEach((cb) => {
        cb.addEventListener("change", () => {
          const r = cb.getAttribute("data-kg-rel");
          if (cb.checked) state.activeRelations.add(r);
          else state.activeRelations.delete(r);
          rebuild();
        });
      });

      // Cap the degree slider to the max degree we actually see.
      const idSet = new Set(state.raw.nodes.map((n) => n.id));
      const degree = computeDegree(state.raw.edges, idSet);
      const maxDeg = Math.max(0, ...degree.values());
      ui.degree.max = String(Math.max(maxDeg, 1));
      ui.degree.value = "0";
      ui.degreeValue.textContent = "0";
    }

    function setMeta(payload) {
      const s = payload.stats || {};
      const total = s.total_nodes ?? (payload.nodes || []).length;
      const edges = s.total_edges ?? (payload.edges || []).length;
      ui.meta.innerHTML = `
        <div><strong>${total}</strong> nodes &middot; <strong>${edges}</strong> edges</div>
        <div>Updated ${escapeHTML(formatTimestamp(payload.generated_at))}</div>
      `;
    }

    function showDetails(nodeId) {
      const node = state.raw.nodes.find((n) => n.id === nodeId);
      if (!node) return;
      ui.details.hidden = false;
      ui.detailsTitle.textContent = node.label || node.id;
      ui.detailsType.textContent = node.type;

      const propEntries = Object.entries(node.properties || {})
        .concat(
          Object.entries(node).filter(
            ([k]) => !["id", "type", "label", "properties"].includes(k),
          ),
        )
        .filter(([, v]) => v != null && v !== "");

      ui.detailsProps.innerHTML = propEntries
        .map(
          ([k, v]) => `
            <div class="kg-prop">
              <dt>${escapeHTML(k)}</dt>
              <dd>${escapeHTML(typeof v === "object" ? JSON.stringify(v) : v)}</dd>
            </div>
          `,
        )
        .join("");

      const actions = [];
      const mediaId = (node.properties && node.properties.media_id) || node.media_id;
      if (node.type === "Media" && mediaId) {
        actions.push(
          `<a href="https://www.morphosource.org/concern/media/${encodeURIComponent(
            mediaId,
          )}" target="_blank" rel="noopener">Open on MorphoSource &rarr;</a>`,
        );
      }
      const specimenId =
        (node.properties && node.properties.specimen_id) || node.specimen_id;
      if (node.type === "Specimen" && specimenId) {
        actions.push(
          `<a href="https://www.morphosource.org/concern/physical_objects/${encodeURIComponent(
            specimenId,
          )}" target="_blank" rel="noopener">Open specimen on MorphoSource &rarr;</a>`,
        );
      }
      actions.push(
        `<a class="kg-secondary" href="#" id="kgFocusNode">Focus this node</a>`,
      );
      ui.detailsActions.innerHTML = actions.join("");

      const focusLink = document.getElementById("kgFocusNode");
      if (focusLink) {
        focusLink.addEventListener("click", (ev) => {
          ev.preventDefault();
          if (state.network) {
            state.network.focus(nodeId, { scale: 1.2, animation: true });
          }
        });
      }
    }

    function buildRunPicker(manifest) {
      const runs = (manifest && manifest.runs) || [];
      const opts = [
        `<option value="${dataUrl("knowledge_graph.json")}">Cumulative (latest)</option>`,
      ];
      for (const r of runs.slice().reverse()) {
        const label = `${formatTimestamp(r.generated_at)} — ${r.run_id || r.file}`;
        opts.push(
          `<option value="${dataUrl(`runs/${r.file}`)}">${escapeHTML(label)}</option>`,
        );
      }
      ui.runPicker.innerHTML = opts.join("");
    }

    async function loadAndRender(url) {
      const payload = await fetchJSON(url, {
        nodes: [],
        edges: [],
        stats: {},
        generated_at: null,
      });
      state.raw = {
        nodes: payload.nodes || [],
        edges: payload.edges || [],
        stats: payload.stats || {},
        generated_at: payload.generated_at,
      };
      buildFilters();
      setMeta(payload);

      if (!state.network) {
        const data = { nodes: state.nodes, edges: state.edges };
        state.network = new vis.Network(ui.graph, data, buildGraphOptions());
        state.network.on("click", (params) => {
          if (params.nodes.length > 0) {
            showDetails(params.nodes[0]);
          } else {
            ui.details.hidden = true;
          }
        });
      }
      rebuild();
    }

    // Event wiring.
    ui.search.addEventListener("input", (ev) => {
      state.search = ev.target.value;
      rebuild();
    });
    ui.degree.addEventListener("input", (ev) => {
      state.minDegree = parseInt(ev.target.value, 10) || 0;
      ui.degreeValue.textContent = String(state.minDegree);
      rebuild();
    });
    ui.fitBtn.addEventListener("click", () => {
      if (state.network) state.network.fit({ animation: true });
    });
    ui.resetBtn.addEventListener("click", () => {
      ui.search.value = "";
      ui.degree.value = "0";
      ui.degreeValue.textContent = "0";
      state.search = "";
      state.minDegree = 0;
      ui.typeFilters.querySelectorAll("input[type=checkbox]").forEach((cb) => {
        cb.checked = true;
        state.activeTypes.add(cb.getAttribute("data-kg-type"));
      });
      ui.relationFilters.querySelectorAll("input[type=checkbox]").forEach((cb) => {
        cb.checked = true;
        state.activeRelations.add(cb.getAttribute("data-kg-rel"));
      });
      rebuild();
      if (state.network) state.network.fit({ animation: true });
    });
    ui.runPicker.addEventListener("change", (ev) => {
      loadAndRender(ev.target.value);
    });
    ui.detailsClose.addEventListener("click", () => {
      ui.details.hidden = true;
    });

    // Initial fetch: manifest first (for the run picker), then the cumulative graph.
    fetchJSON(dataUrl("runs/_manifest.json"), { runs: [] }).then((manifest) => {
      buildRunPicker(manifest);
      loadAndRender(state.currentDataUrl);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
