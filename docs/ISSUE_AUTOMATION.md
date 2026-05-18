# Issue Automation (IssueOps)

This repository runs an **IssueOps-style automation system** for
metadata-to-MorphoSource comparison and verification requests. The pattern is
adapted from
[`MorphoCloud/MorphoCloudInstances`](https://github.com/MorphoCloud/MorphoCloudInstances/tree/main/.github/actions),
trimmed of its OpenStack-specific machinery, and reshaped around the
comparison/verification workflows in this repo.

The headline behavior:

- A user opens a **Comparison Request** issue from the YAML form template.
- A workflow auto-applies labels (`mode:*`, `input:*`, `status:awaiting-approval`)
  and posts a **verification progress checklist** comment with emoji
  status markers (`⏳` in progress, `✅` done, `❌` failed, `⚪` skipped).
- A maintainer comments `/approve` to unlock the pipeline.
- The issue author (or a maintainer) comments `/verify` to run the comparison.
  Each stage updates the same checklist comment in place, so the issue is a
  live status board.

## Files at a glance

| Path                                                      | Role                                                                  |
| --------------------------------------------------------- | --------------------------------------------------------------------- |
| `.github/ISSUE_TEMPLATE/comparison_request.yml`           | Structured form template that gives every request the same fields.   |
| `.github/verification-progress-template.md`               | Markdown template rendered into the progress checklist comment.       |
| `.github/issue-commands.md`                               | Help text listing supported slash commands; auto-posted on each issue. |
| `.github/actions/extract-issue-fields/action.yml`         | Parses the form into per-field outputs (uses `stefanbuck/github-issue-parser`). |
| `.github/actions/comment-progress/action.yml`             | Renders + posts/updates the emoji progress checklist.                 |
| `.github/actions/check-approval/action.yml`               | Returns `true/false` for the `request:verified` label.                |
| `.github/actions/update-approval/action.yml`              | Toggles `request:verified` and posts a confirmation comment.          |
| `.github/actions/update-status-label/action.yml`          | Sets exactly one `status:*` label on the issue.                       |
| `.github/workflows/labels.yml`                            | Source of truth for repo labels; runs on edit + `workflow_dispatch`.  |
| `.github/workflows/on-request-opened.yml`                 | Fan-out workflow triggered by `issues: opened`.                       |
| `.github/workflows/request-labeler.yml`                   | Adds `mode:*`/`input:*` labels and rotates `status:*`.                |
| `.github/workflows/request-initial-comments.yml`          | Posts the progress checklist + commands help comment.                 |
| `.github/workflows/request-notify-admin.yml`              | Pings maintainers (and emails them if SMTP creds are present).        |
| `.github/workflows/issue-commands.yml`                    | Slash-command dispatcher for `/approve`, `/unapprove`, `/verify`.     |
| `.github/workflows/on-admin-mention.yml`                  | Sends a notification when the admin handle is `@`-mentioned.          |
| `.github/workflows/verify-comparison.yml`                 | Runs the verification pipeline; live-updates progress emojis.         |

## Lifecycle of a comparison request

```
issues: opened
        │
        ▼
┌──────────────────────┐   ┌─────────────────────────────┐   ┌────────────────────────┐
│ on-request-opened    │──▶│ request-labeler             │   │ request-initial-comments│
└──────────────────────┘   │  (mode:*, input:*,          │   │  posts ⏳ checklist     │
        │                  │   status:awaiting-approval) │   │  + slash-commands help │
        │                  └─────────────────────────────┘   └────────────────────────┘
        ▼
┌──────────────────────┐
│ request-notify-admin │  → comment + optional email
└──────────────────────┘

issue_comment: created
        │
        ▼
┌──────────────────────┐
│ issue-commands       │  /approve, /unapprove, /verify, /help
└────────┬─────────────┘
         │  (when /verify is approved)
         ▼
┌──────────────────────┐
│ verify-comparison    │  parse fields → run compare/verify → publish report
│  status:running →    │  updates the same progress checklist comment in place
│  status:completed/   │
│  status:failed       │
└──────────────────────┘
```

## Slash commands

| Command           | Allowed for                                | Effect                                                                |
| ----------------- | ------------------------------------------ | --------------------------------------------------------------------- |
| `/approve`        | Maintainers (`triage`+)                    | Adds `request:verified`; unlocks `/verify` for the issue author.      |
| `/unapprove`      | Maintainers (`triage`+)                    | Removes `request:verified`.                                           |
| `/verify`         | Maintainers OR issue author (post-approval) | Triggers `verify-comparison.yml` for the issue.                       |
| `/run-comparison` | Same as `/verify`                          | Alias for `/verify`.                                                  |
| `/help`           | Anyone                                     | Re-posts the issue-commands help comment.                             |

The dispatcher reacts with `🚀` to the comment that issued an accepted command,
and posts a follow-up comment on success or failure.

## Labels

`labels.yml` is the single source of truth. To change a color, description, or
add a new label, edit that file and merge to `main` — a workflow run will
reconcile the repo labels via `gh label create --force`. Categories:

- `request-type:*` — kind of issue (currently `comparison`, `bug`, `feature`).
- `request-creator:*` — `user` or `bot`.
- `request:*` — gating labels (`verified`, `invalid`, `duplicate`).
- `mode:*` — extracted comparison mode.
- `input:*` — extracted input source.
- `status:*` — exclusive lifecycle state, rotated by `update-status-label`.

## Required configuration

The system works out of the box with just `GITHUB_TOKEN`. The optional
maintainer-mention email path uses these repository settings:

| Kind     | Name             | Purpose                                                              |
| -------- | ---------------- | -------------------------------------------------------------------- |
| Variable | `ADMIN_HANDLE`   | Handle to ping (defaults to `@johntrue15`). Use a team for multi-user. |
| Variable | `ADMIN_EMAILS`   | Comma-separated email recipients for admin notifications.            |
| Secret   | `MAIL_USERNAME`  | SMTP username (Gmail in the default action config).                  |
| Secret   | `MAIL_PASSWORD`  | SMTP password / app password.                                        |

If `ADMIN_EMAILS` or the SMTP secrets are missing, the email steps are
silently skipped — the in-issue comment ping still fires.

## Extending

- **New comparison mode** — add the option to `comparison_request.yml`, the
  `mode:*` label to `labels.yml`, the case-statement to `request-labeler.yml`,
  and the dispatch branch to `verify-comparison.yml`.
- **New slash command** — add a case in `issue-commands.yml`'s detection step,
  a permission check (if needed), and a step that performs the action.
- **New verification stage** — add an input to
  `.github/actions/comment-progress/action.yml`, a column in
  `verification-progress-template.md`, and update the calls in
  `request-initial-comments.yml`, `request-notify-admin.yml`, and
  `verify-comparison.yml`.

## Provenance

The skeleton (composite-action layout, the emoji progress comment idea, the
`status:*` label rotator, and the slash-command dispatcher) follows
[`MorphoCloud/MorphoCloudInstances/.github/actions`](https://github.com/MorphoCloud/MorphoCloudInstances/tree/main/.github/actions).
What's specific to this repo is the issue template, the `mode:*` /
`input:*` taxonomy, and the `verify-comparison.yml` pipeline that bridges
into `compare.py` / `verify_pixel_spacing.py`.
