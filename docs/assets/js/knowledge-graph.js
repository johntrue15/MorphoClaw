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
 * any page that doesn't include the #kg-app container. vis-network itself
 * is loaded lazily from a locally-vendored bundle (with a CDN fallback)
 * so this script also works on pages served from a project sub-path.
 */

(function () {
  "use strict";

  // Captured at script-parse time so it survives async work below.
  // document.currentScript is the <script src="..."> tag MkDocs emitted
  // for assets/js/knowledge-graph.js, so its .src is an absolute URL we
  // can use to derive sibling asset URLs without hard-coding any path.
  const SELF_SCRIPT_URL = (function () {
    try {
      if (document.currentScript && document.currentScript.src) {
        return document.currentScript.src;
      }
    } catch (_) { /* ignore */ }
    return null;
  })();

  const VIS_NETWORK_CDN =
    "https://cdn.jsdelivr.net/npm/vis-network@9.1.6/standalone/umd/vis-network.min.js";

  // Performance budget. Above ~2k visible nodes vis-network's physics engine
  // becomes unusable on most browsers, so we cap what we feed to it and
  // sample the highest-degree subset. Users can opt in to more via a slider.
  const NODE_CAP_DEFAULT = 1500;
  const NODE_CAP_MAX = 5000;
  // Above this threshold we don't run an ongoing physics simulation; we
  // stabilise once and freeze, which renders large graphs in a few seconds
  // instead of locking the tab for 30 seconds.
  const PHYSICS_AUTO_THRESHOLD = 800;

  function vendoredVisNetworkUrl() {
    if (!SELF_SCRIPT_URL) return null;
    try {
      // SELF_SCRIPT_URL looks like .../assets/js/knowledge-graph.js
      // We want   .../assets/vendor/vis-network/vis-network.min.js
      return new URL(
        "../vendor/vis-network/vis-network.min.js",
        SELF_SCRIPT_URL
      ).toString();
    } catch (_) {
      return null;
    }
  }

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

  // Resolve URLs against the site root, derived from this script's own URL.
  // This script lives at <root>/assets/js/knowledge-graph.js, so two "../"
  // jumps put us at the docs site root regardless of which page is loading
  // us. Falls back to a window.location-relative URL for non-browser tests.
  function siteRootUrl(relative) {
    const cleaned = String(relative || "").replace(/^\/+/, "");
    if (SELF_SCRIPT_URL) {
      return new URL("../../" + cleaned, SELF_SCRIPT_URL).toString();
    }
    const base = new URL(".", window.location.href);
    return new URL(cleaned, base).toString();
  }

  function dataUrl(name) {
    const u = new URL(siteRootUrl(`data/${name}`));
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
      nodeCap: document.getElementById("kgNodeCap"),
      nodeCapValue: document.getElementById("kgNodeCapValue"),
      physics: document.getElementById("kgPhysics"),
      status: document.getElementById("kgStatus"),
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
        "<p>The interactive viewer requires the bundled <code>vis-network</code> script. " +
        "The local copy at <code>assets/vendor/vis-network/vis-network.min.js</code> and the " +
        "jsDelivr fallback both failed to load &mdash; check the browser network tab for the " +
        "actual error (CSP, ad-blocker, or offline). You can still download the raw graph " +
        "from <a href=\"" + dataUrl("knowledge_graph.json") + "\">data/knowledge_graph.json</a>.</p>";
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
      // Performance knobs (user-tunable from the sidebar).
      nodeCap: NODE_CAP_DEFAULT,
      // 'auto' | 'on' | 'off'. In 'auto' we enable physics only for small
      // graphs so the cumulative view doesn't lock the browser.
      physicsMode: "auto",
      // Caches of the original payload so filters can rebuild without re-fetching.
      raw: { nodes: [], edges: [], stats: {}, generated_at: null },
      nodeDegree: new Map(),
      // Last-render diagnostics surfaced in the status banner.
      lastRender: { rendered: 0, total: 0, sampled: false, physics: false },
    };

    function shouldUsePhysics(nodeCount) {
      if (state.physicsMode === "on") return true;
      if (state.physicsMode === "off") return false;
      return nodeCount <= PHYSICS_AUTO_THRESHOLD;
    }

    function buildGraphOptions(renderedNodes) {
      const usePhysics = shouldUsePhysics(renderedNodes);
      // Stabilisation budget scales inversely with node count. forceAtlas2
      // is O(N^2) per tick, so we drop iterations sharply at scale.
      const stabIters = usePhysics
        ? Math.max(50, Math.min(200, Math.floor(20000 / Math.max(1, renderedNodes))))
        : 0;
      return {
        physics: usePhysics
          ? {
              enabled: true,
              solver: "forceAtlas2Based",
              forceAtlas2Based: {
                gravitationalConstant: -30,
                springLength: 80,
                springConstant: 0.04,
              },
              stabilization: {
                enabled: true,
                iterations: stabIters,
                updateInterval: 25,
                fit: true,
              },
              adaptiveTimestep: true,
              maxVelocity: 30,
            }
          : { enabled: false },
        layout: usePhysics
          ? {}
          : {
              // Static layout helps vis-network skip the brutal physics
              // bootstrap when we've turned physics off. Improved layout
              // gives it a reasonable initial placement in one pass.
              improvedLayout: true,
              randomSeed: 7,
            },
        interaction: {
          hover: true,
          tooltipDelay: 150,
          zoomView: true,
          dragView: true,
          navigationButtons: false,
        },
        edges: {
          smooth: usePhysics ? { type: "continuous" } : false,
          color: { color: "#8b949e", opacity: 0.4 },
          font: { size: 9, color: "#8b949e", strokeWidth: 0 },
          width: 0.6,
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

      // Step 1: collect everything that survives the user-side filters.
      const candidates = [];
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
        candidates.push({ node: n, degree: d });
      }

      // Step 2: enforce the node budget. We keep the highest-degree nodes
      // so the rendered subgraph is the most connected slice. Search-hit
      // nodes are always kept first (boosted) so the search remains useful
      // even when its results are low-degree.
      const cap = Math.max(50, Math.min(NODE_CAP_MAX, state.nodeCap | 0));
      let sampled = false;
      let kept = candidates;
      if (candidates.length > cap) {
        sampled = true;
        const isHit = q
          ? (n) =>
              (n.label || "").toLowerCase().includes(q) ||
              (n.id || "").toLowerCase().includes(q)
          : () => false;
        candidates.sort((a, b) => {
          const ah = isHit(a.node) ? 1 : 0;
          const bh = isHit(b.node) ? 1 : 0;
          if (ah !== bh) return bh - ah;
          return b.degree - a.degree;
        });
        kept = candidates.slice(0, cap);
      }

      const visibleNodeIds = new Set();
      const visNodes = [];
      for (const { node: n } of kept) {
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

      // Apply physics options appropriate for the *current* render size.
      // This is the single most important defence against the cumulative
      // graph locking the browser: when rebuild() finds 4500 visible
      // nodes, we flip physics off automatically.
      const usePhysics = shouldUsePhysics(visNodes.length);
      if (state.network) {
        state.network.setOptions(buildGraphOptions(visNodes.length));
      }

      // Use update-in-place batch APIs. clear+add is far slower for vis
      // DataSets in the multi-thousand-node case because each emits its
      // own change event.
      state.nodes.clear();
      state.edges.clear();
      state.nodes.add(visNodes);
      state.edges.add(visEdges);

      state.lastRender = {
        rendered: visNodes.length,
        total: candidates.length,
        sampled,
        physics: usePhysics,
      };
      setStatus();

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

    function setStatus() {
      if (!ui.status) return;
      const { rendered, total, sampled, physics } = state.lastRender;
      if (rendered === 0 && total === 0) {
        ui.status.hidden = true;
        return;
      }
      const physicsLabel = physics ? "physics: on" : "physics: off";
      if (sampled) {
        ui.status.hidden = false;
        ui.status.className = "kg-status kg-status-warn";
        ui.status.innerHTML =
          `Showing <strong>${rendered.toLocaleString()}</strong> of ` +
          `<strong>${total.toLocaleString()}</strong> matching nodes ` +
          `(highest-degree subset, ${physicsLabel}). ` +
          `Use filters or raise the node cap to see more.`;
      } else {
        ui.status.hidden = false;
        ui.status.className = "kg-status";
        ui.status.innerHTML =
          `Rendering <strong>${rendered.toLocaleString()}</strong> nodes (${physicsLabel}).`;
      }
    }

    function buildFilters() {
      const types = new Map();
      for (const n of state.raw.nodes) {
        types.set(n.type, (types.get(n.type) || 0) + 1);
      }
      const sortedTypes = [...types.keys()].sort();
      state.activeTypes = new Set(sortedTypes);
      // autocomplete="off" prevents Safari/Firefox from restoring stale
      // checked-state when innerHTML re-renders the filter group; without
      // it the previous session's interaction can leave checkboxes
      // visually unchecked while state.activeTypes still says they're on.
      ui.typeFilters.innerHTML = sortedTypes
        .map(
          (t) => `
            <label class="kg-checkbox">
              <input type="checkbox" data-kg-type="${escapeHTML(t)}" autocomplete="off" checked />
              <span class="kg-swatch" style="background:${TYPE_COLORS[t] || "#8b949e"}"></span>
              <span>${escapeHTML(t)}</span>
              <span class="kg-count" data-count="${escapeHTML(t)}">${types.get(t)}</span>
            </label>
          `,
        )
        .join("");
      // Force the DOM state to match what we just wrote. The `checked`
      // attribute alone isn't enough on browsers that aggressively
      // restore form values across reloads.
      ui.typeFilters.querySelectorAll("input[type=checkbox]").forEach((cb) => {
        cb.checked = true;
      });

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
              <input type="checkbox" data-kg-rel="${escapeHTML(r)}" autocomplete="off" checked />
              <span>${escapeHTML(r)}</span>
              <span class="kg-count">${rels.get(r)}</span>
            </label>
          `,
        )
        .join("");
      ui.relationFilters.querySelectorAll("input[type=checkbox]").forEach((cb) => {
        cb.checked = true;
      });

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

    function shortTopic(topic, max = 70) {
      if (!topic) return "";
      const trimmed = topic.trim().replace(/\s+/g, " ");
      return trimmed.length > max ? trimmed.slice(0, max - 1) + "…" : trimmed;
    }

    function buildRunPicker(manifest) {
      const runs = (manifest && manifest.runs) || [];
      const opts = [];

      // Individual runs first (most recent at the top), each with its
      // node count baked into the label so users can pick a manageable
      // size at a glance.
      for (const r of runs.slice().reverse()) {
        const topic = shortTopic(r.topic) || r.file;
        const stats = r.stats || {};
        const nNodes = stats.total_nodes || 0;
        const sizeHint = nNodes ? ` · ${nNodes.toLocaleString()} nodes` : "";
        const label = `${formatTimestamp(r.generated_at)} — ${topic}${sizeHint}`;
        const titleAttr = escapeHTML(
          r.topic ||
            r.run_id ||
            r.file ||
            "(unknown topic)",
        );
        opts.push(
          `<option value="${dataUrl(`runs/${r.file}`)}" ` +
            `data-run-file="${escapeHTML(r.file)}" ` +
            `data-topic="${escapeHTML(r.topic || "")}" ` +
            `data-size="${nNodes}" ` +
            `title="${titleAttr}">${escapeHTML(label)}</option>`,
        );
      }

      // Cumulative goes LAST in the dropdown and is flagged as the
      // performance-sensitive option, so it's an explicit user choice
      // rather than the default landing experience.
      opts.push(
        `<option value="${dataUrl("knowledge_graph.json")}" ` +
          `data-run-file="" data-topic="Cumulative" data-size="cumulative">` +
          `— Cumulative across all runs (slow)</option>`,
      );

      ui.runPicker.innerHTML = opts.join("");
    }

    function preferredInitialUrl(manifest) {
      // The cumulative graph can be 8000+ nodes which locks vis-network's
      // physics on first paint. Default to the most recent individual run
      // instead. Users can pick "Cumulative" from the picker if they
      // explicitly want the union view.
      const runs = (manifest && manifest.runs) || [];
      if (runs.length > 0) {
        // _manifest.runs is ordered oldest -> newest; take the last entry.
        const latest = runs[runs.length - 1];
        return dataUrl(`runs/${latest.file}`);
      }
      return dataUrl("knowledge_graph.json");
    }

    function selectInitialRun(manifest) {
      // Honour ?run=<file> in the URL so example-query deep links from the
      // submission page jump straight to the relevant snapshot. Falls back
      // to the most recent individual run (NOT cumulative) so the page
      // never lands on a multi-thousand-node graph the user didn't ask for.
      let target = preferredInitialUrl(manifest);
      try {
        const params = new URLSearchParams(window.location.search);
        const wanted = (params.get("run") || "").trim();
        if (wanted) {
          if (wanted.toLowerCase() === "cumulative") {
            target = dataUrl("knowledge_graph.json");
          } else {
            const runs = (manifest && manifest.runs) || [];
            const lowered = wanted.toLowerCase();
            const hit = runs.find(
              (r) =>
                r.file === wanted ||
                (r.topic && r.topic.toLowerCase().includes(lowered)) ||
                (r.local_run_id &&
                  r.local_run_id.toLowerCase().includes(lowered)) ||
                (r.run_id && r.run_id.toLowerCase().includes(lowered)),
            );
            if (hit) {
              target = dataUrl(`runs/${hit.file}`);
            } else {
              console.warn("[KG] ?run=%s did not match any snapshot.", wanted);
            }
          }
        }
        // Pre-select the matching option once the picker is built.
        const pickerVal = target;
        setTimeout(() => {
          const opt = Array.from(ui.runPicker.options).find(
            (o) => o.value === pickerVal,
          );
          if (opt) ui.runPicker.value = pickerVal;
        }, 0);
      } catch (err) {
        console.warn("[KG] Could not parse URL params:", err);
      }
      state.currentDataUrl = target;
      return target;
    }

    async function loadAndRender(url) {
      console.info("[KG] Loading graph data:", url);

      // Show a loading hint immediately so the UI doesn't appear frozen
      // while we fetch the (potentially MB-sized) JSON.
      if (ui.status) {
        ui.status.hidden = false;
        ui.status.className = "kg-status";
        ui.status.textContent = "Loading snapshot…";
      }
      if (ui.meta) {
        ui.meta.textContent = "Loading…";
      }
      state.currentDataUrl = url;

      const t0 = performance.now();
      const payload = await fetchJSON(url, {
        nodes: [],
        edges: [],
        stats: {},
        generated_at: null,
      });
      const fetchMs = (performance.now() - t0).toFixed(0);
      console.info(
        "[KG] Loaded %d nodes / %d edges in %sms (stats: %o)",
        (payload.nodes || []).length,
        (payload.edges || []).length,
        fetchMs,
        payload.stats || {},
      );

      // Hide the empty-state pre-emptively when we know real data came
      // back; rebuild() will toggle it again if the active filter set
      // happens to filter every node out.
      if (ui.empty) {
        ui.empty.hidden = (payload.nodes || []).length > 0;
      }
      state.raw = {
        nodes: payload.nodes || [],
        edges: payload.edges || [],
        stats: payload.stats || {},
        generated_at: payload.generated_at,
      };
      buildFilters();
      setMeta(payload);

      // Re-create the vis.Network for each new dataset so physics options
      // and DataSet bindings start clean. Reusing the old instance with
      // setOptions(...) is fine for filter-driven rebuilds, but a fresh
      // dataset deserves a fresh stabilisation pass.
      if (state.network) {
        state.network.destroy();
        state.network = null;
        // Reset DataSets so the new network doesn't inherit stale nodes.
        state.nodes = new vis.DataSet();
        state.edges = new vis.DataSet();
      }
      const data = { nodes: state.nodes, edges: state.edges };
      // We pre-compute estimated render size to choose physics correctly
      // on first paint. The actual render goes through rebuild() below.
      const estimate = Math.min(
        state.nodeCap,
        (payload.nodes || []).length || 0,
      );
      state.network = new vis.Network(ui.graph, data, buildGraphOptions(estimate));
      state.network.on("click", (params) => {
        if (params.nodes.length > 0) {
          showDetails(params.nodes[0]);
        } else {
          ui.details.hidden = true;
        }
      });
      state.network.once("stabilizationIterationsDone", () => {
        console.info("[KG] Stabilisation complete.");
      });

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
      state.nodeCap = NODE_CAP_DEFAULT;
      if (ui.nodeCap) ui.nodeCap.value = String(NODE_CAP_DEFAULT);
      if (ui.nodeCapValue)
        ui.nodeCapValue.textContent = String(NODE_CAP_DEFAULT);
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
      const opt = ev.target.options[ev.target.selectedIndex];
      const isCumulative =
        opt && opt.getAttribute("data-size") === "cumulative";
      if (isCumulative) {
        // Confirm before loading the cumulative view — without this an
        // accidental click could lock up the tab for tens of seconds on
        // slower devices.
        const ok = window.confirm(
          "The cumulative graph contains ~8,000 nodes from every run.\n\n" +
            "It will be sampled to the current node-cap (" +
            state.nodeCap +
            ") for performance. Continue?",
        );
        if (!ok) {
          // Revert the picker to whatever was active before.
          ev.target.value = state.currentDataUrl;
          return;
        }
      }
      loadAndRender(ev.target.value);
    });

    if (ui.nodeCap) {
      ui.nodeCap.min = "100";
      ui.nodeCap.max = String(NODE_CAP_MAX);
      ui.nodeCap.step = "100";
      ui.nodeCap.value = String(state.nodeCap);
      if (ui.nodeCapValue)
        ui.nodeCapValue.textContent = String(state.nodeCap);
      ui.nodeCap.addEventListener("input", (ev) => {
        state.nodeCap = parseInt(ev.target.value, 10) || NODE_CAP_DEFAULT;
        if (ui.nodeCapValue)
          ui.nodeCapValue.textContent = String(state.nodeCap);
        rebuild();
      });
    }
    if (ui.physics) {
      ui.physics.value = state.physicsMode;
      ui.physics.addEventListener("change", (ev) => {
        state.physicsMode = ev.target.value || "auto";
        // Re-create the network so the new physics setting applies to the
        // initial layout, not just future drag events.
        loadAndRender(state.currentDataUrl);
      });
    }
    ui.detailsClose.addEventListener("click", () => {
      ui.details.hidden = true;
    });

    // Initial fetch: manifest first (for the run picker), then the chosen
    // snapshot (deep-linked via ?run=… when present, else the cumulative).
    fetchJSON(dataUrl("runs/_manifest.json"), { runs: [] }).then((manifest) => {
      buildRunPicker(manifest);
      const initial = selectInitialRun(manifest);
      loadAndRender(initial);
    });
  }

  // ---------------------------------------------------------------------------
  // vis-network loader with CDN fallback.
  //
  // We don't depend on the page including an inline <script> tag for
  // vis-network. Instead we compute the locally-vendored URL relative to
  // *this* script and inject it ourselves. If that fails (network error,
  // 404 during a partial deploy, etc.) we fall back to a public CDN.
  // ---------------------------------------------------------------------------

  function loadScript(src, { crossOrigin = false } = {}) {
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src;
      s.async = false;
      if (crossOrigin) s.crossOrigin = "anonymous";
      s.onload = () => resolve(src);
      s.onerror = () => reject(new Error(`Failed to load ${src}`));
      document.head.appendChild(s);
    });
  }

  async function ensureVisNetwork() {
    if (typeof vis !== "undefined" && vis.Network) return true;

    const localUrl = vendoredVisNetworkUrl();
    if (localUrl) {
      try {
        console.info("[KG] Loading vis-network from vendored bundle:", localUrl);
        await loadScript(localUrl);
        if (typeof vis !== "undefined" && vis.Network) return true;
      } catch (err) {
        console.warn("[KG] Vendored vis-network unreachable:", err);
      }
    }

    try {
      console.warn("[KG] Falling back to CDN vis-network:", VIS_NETWORK_CDN);
      await loadScript(VIS_NETWORK_CDN, { crossOrigin: true });
    } catch (err) {
      console.error("[KG] CDN vis-network fallback failed:", err);
      return false;
    }
    return typeof vis !== "undefined" && !!vis.Network;
  }

  function boot() {
    // Only do work on pages that actually host the viewer.
    if (!document.getElementById("kg-app")) return;
    ensureVisNetwork().then(init);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
