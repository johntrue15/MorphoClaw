---
title: Submit a Query
description: Ask AutoResearchClaw a question about MorphoSource specimens — the answer comes back as a comment on a GitHub issue.
hide:
  - toc
---

# Submit a Query

Type a natural-language question below. Clicking **Prepare to Submit Query**
creates a pre-filled GitHub issue with the `query-request` label; an automation
workflow picks it up, runs your search against MorphoSource, and posts results
back as a comment.

<div class="grid cards" markdown>

- :material-information-outline: **What you need** &mdash; a free GitHub account.

- :material-clock-fast: **Typical latency** &mdash; 1–2 minutes from issue creation to result comment.

- :material-shield-check: **Where results live** &mdash; on the issue you just created (you'll be subscribed for notifications).

</div>

<div class="arc-query" markdown>

<form id="queryForm" class="arc-query-form" autocomplete="off">
  <label for="queryText">Enter your query</label>
  <textarea id="queryText" name="query" placeholder="Example: Tell me about lizard specimens with CT scans..." required></textarea>

  <div class="arc-query-buttons">
    <button type="submit" class="arc-btn arc-btn-primary" id="submitBtn">
      Prepare to Submit Query
    </button>
    <button type="button" class="arc-btn arc-btn-secondary" onclick="arcQueryClear()">
      Clear
    </button>
  </div>
</form>

<div class="arc-examples">
  <strong>Example queries (click to insert):</strong>
  <ul>
    <li onclick="arcQuerySet('Tell me about lizard specimens')">Tell me about lizard specimens</li>
    <li onclick="arcQuerySet('How many snake specimens are available?')">How many snake specimens are available?</li>
    <li onclick="arcQuerySet('Show me CT scans of crocodiles')">Show me CT scans of crocodiles</li>
    <li onclick="arcQuerySet('Compare cranial morphology across primate species')">Compare cranial morphology across primate species</li>
  </ul>
</div>

<div id="status" class="arc-status" hidden></div>
<div id="workflowLink" class="arc-workflow" hidden></div>

</div>

<style>
.arc-query {
  background: var(--md-default-bg-color);
  border: 1px solid var(--md-default-fg-color--lightest);
  border-radius: 12px;
  padding: 1.5rem;
  margin: 1.5rem 0;
}
.arc-query-form label {
  display: block;
  font-weight: 600;
  margin-bottom: 0.5rem;
}
.arc-query-form textarea {
  width: 100%;
  min-height: 120px;
  padding: 0.85rem;
  border: 1px solid var(--md-default-fg-color--lightest);
  border-radius: 8px;
  background: var(--md-code-bg-color);
  color: var(--md-default-fg-color);
  font: inherit;
  resize: vertical;
}
.arc-query-form textarea:focus {
  outline: none;
  border-color: var(--md-primary-fg-color);
}
.arc-query-buttons {
  display: flex;
  gap: 0.75rem;
  margin-top: 1rem;
}
.arc-query-buttons .arc-btn {
  flex: 1;
}
.arc-examples {
  margin-top: 1.25rem;
  padding: 1rem;
  background: var(--md-code-bg-color);
  border-radius: 8px;
  font-size: 0.9rem;
}
.arc-examples ul {
  list-style: none;
  padding: 0;
  margin: 0.5rem 0 0 0;
}
.arc-examples li {
  padding: 0.5rem 0.75rem;
  margin: 0.25rem 0;
  background: var(--md-default-bg-color);
  border: 1px solid var(--md-default-fg-color--lightest);
  border-radius: 6px;
  cursor: pointer;
  transition: transform 0.15s ease, background 0.15s ease;
}
.arc-examples li:hover {
  transform: translateX(3px);
  background: var(--md-primary-fg-color);
  color: var(--md-primary-bg-color);
}
.arc-status {
  margin-top: 1rem;
  padding: 1rem;
  border-radius: 8px;
}
.arc-status.success { background: rgba(63, 185, 80, 0.12); border: 1px solid rgba(63, 185, 80, 0.4); color: var(--md-default-fg-color); }
.arc-status.error   { background: rgba(248, 81, 73, 0.12); border: 1px solid rgba(248, 81, 73, 0.4); color: var(--md-default-fg-color); }
.arc-workflow {
  margin-top: 1rem;
  padding: 1rem;
  background: var(--md-code-bg-color);
  border-radius: 8px;
}
.arc-workflow a {
  font-weight: 600;
}
</style>

<script>
(function () {
  const GITHUB_OWNER = "johntrue15";
  const GITHUB_REPO = "Metadata-to-Morphsource-compare";
  const NEW_ISSUE_URL = `https://github.com/${GITHUB_OWNER}/${GITHUB_REPO}/issues/new`;
  const WORKFLOW_URL  = `https://github.com/${GITHUB_OWNER}/${GITHUB_REPO}/actions/workflows/query-processor.yml`;

  const form = document.getElementById("queryForm");
  if (!form) return;
  const statusDiv = document.getElementById("status");
  const workflowLinkDiv = document.getElementById("workflowLink");
  const submitBtn = document.getElementById("submitBtn");

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = document.getElementById("queryText").value.trim();
    if (!text) return;

    submitBtn.disabled = true;
    submitBtn.textContent = "Preparing…";

    try {
      const title = `Query: ${text.substring(0, 50)}${text.length > 50 ? "…" : ""}`;
      const body = [
        "## MorphoSource Query Submission",
        "",
        "**Query:**",
        text,
        "",
        "---",
        "*Submitted via the AutoResearchClaw docs site. A GitHub Actions workflow will process this query and post results as a comment.*",
      ].join("\n");

      const params = new URLSearchParams({ title, body, labels: "query-request" });
      const issueUrl = `${NEW_ISSUE_URL}?${params.toString()}`;

      statusDiv.hidden = false;
      statusDiv.className = "arc-status success";
      statusDiv.innerHTML =
        "<strong>Ready to submit your query.</strong><br/>" +
        "Open the GitHub issue form below; clicking <em>Submit new issue</em> kicks off the workflow.";

      workflowLinkDiv.hidden = false;
      workflowLinkDiv.innerHTML =
        `<a href="${issueUrl}" target="_blank" rel="noopener">Open the pre-filled GitHub issue &rarr;</a>` +
        `<br/><small>Alternative: <a href="${WORKFLOW_URL}" target="_blank" rel="noopener">manually trigger the workflow &rarr;</a></small>`;
    } catch (err) {
      statusDiv.hidden = false;
      statusDiv.className = "arc-status error";
      statusDiv.textContent = `Could not prepare submission: ${err.message}`;
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Prepare to Submit Query";
    }
  });

  window.arcQuerySet = (text) => {
    document.getElementById("queryText").value = text;
  };
  window.arcQueryClear = () => {
    document.getElementById("queryText").value = "";
    statusDiv.hidden = true;
    workflowLinkDiv.hidden = true;
  };
})();
</script>

## How submission works under the hood

```mermaid
sequenceDiagram
    participant U as You
    participant D as Docs site (this page)
    participant GH as GitHub Issues
    participant WF as on-request-opened.yml
    participant API as MorphoSource API

    U->>D: Type query, click submit
    D->>GH: Pre-fills issue form (label: query-request)
    U->>GH: Confirms issue creation
    GH->>WF: Issue opened event
    WF->>API: query_formatter → MorphoSource search
    API-->>WF: Search results JSON
    WF->>GH: Post results as comment
    GH-->>U: Notification
```

See the [Query System Guide](QUERY_SYSTEM_GUIDE.md) for the full pipeline, the
[Submission Guide](QUERY_SUBMISSION_GUIDE.md) for formatting tips, and the
[Issue Automation](ISSUE_AUTOMATION.md) docs for how labels and routing work.
