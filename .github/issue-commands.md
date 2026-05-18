### Supported Issue Commands

This issue is wired up to a small **IssueOps** workflow. Maintainers and (in
some cases) the issue author can post these slash commands as a comment to
trigger automation. Each command reacts with 🚀 when accepted and posts a
follow-up confirmation or error comment.

| Command            | Who can run it                  | What it does                                                                                          |
| ------------------ | ------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `/approve`         | Maintainers                     | Adds the `request:verified` label, unlocking comparison commands for the issue author.                |
| `/unapprove`       | Maintainers                     | Removes `request:verified` and locks the comparison commands again.                                   |
| `/verify`          | Maintainers, issue author¹      | Runs the metadata-to-MorphoSource verification pipeline against the dataset described in this issue.  |
| `/run-comparison`  | Maintainers, issue author¹      | Alias for `/verify`. Triggers the same pipeline.                                                      |
| `/help`            | Anyone                          | Re-posts this help message.                                                                           |

¹ The author can run these commands only after a maintainer has issued
`/approve`. The `request:verified` label gates the pipeline.

While a command is running, the **Verification progress** comment above is
updated in place. Step indicators move from blank → ⏳ → ✅ (or ❌ if a step
fails). The overall status next to the heading reflects the current state.

If anything looks wrong, mention the maintainers (`@johntrue15`) or open a
companion bug report.
