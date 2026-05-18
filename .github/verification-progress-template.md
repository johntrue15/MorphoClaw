<!--
  Verification progress comment.
  This template is rendered by .github/actions/comment-progress and updated
  by the workflows under .github/workflows/.
  Each `{{ var }}` is replaced with one of: '', '⏳', '✅', '❌', or '⚪'.
-->

### Verification progress {{ status }}

| Step | Status |
| ---- | :----: |
| Validate issue template | {{ validate_template }} |
| Parse fields            | {{ parse_fields }}      |
| Apply labels            | {{ apply_labels }}      |
| Notify admins           | {{ notify_admins }}     |
| Run comparison          | {{ run_comparison }}    |
| Verify results          | {{ verify_results }}    |
| Publish report          | {{ publish_report }}    |

Legend: ⏳ in progress · ✅ done · ❌ failed · ⚪ skipped · _blank_ not yet started

🔗 [Workflow run details]({{ details_url }})
