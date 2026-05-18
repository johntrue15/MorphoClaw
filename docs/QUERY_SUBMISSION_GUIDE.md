# Query Submission Guide

## 🎯 How to Submit a Query

### Step 1: Visit the Query Page
Go to: **[https://johntrue15.github.io/MorphoClaw/](https://johntrue15.github.io/MorphoClaw/)**

### Step 2: Enter Your Question
Type your question in the text box. Examples:
- "Tell me about lizard specimens"
- "How many snake specimens are available?"
- "Show me CT scans of crocodiles"

### Step 3: Click "Prepare to Submit Query"
This will:
- Prepare your query
- Show submission instructions
- Generate a pre-filled GitHub Issue link

### Step 4: Create the GitHub Issue
Click the link to open GitHub's issue creation page. The form will be pre-filled with:
- **Title:** Your query (first 50 characters)
- **Body:** Your full query text
- **Label:** `query-request` (automatically triggers the workflow)

### Step 5: Submit the Issue
Click the green "Submit new issue" button on GitHub.

### Step 6: Wait for Results
- The system automatically processes your query (usually 1-2 minutes)
- You'll receive a GitHub notification when complete
- Results are posted as a comment on your issue
- The issue is automatically closed when processing finishes

## ❓ Frequently Asked Questions

### Do I need a GitHub account?
**Yes**, you need a free GitHub account to submit queries. This is required to create issues.

If you don't want to create an account, you can still use the manual workflow trigger method (see [QUERY_SYSTEM_GUIDE.md](QUERY_SYSTEM_GUIDE.md)).

### Why use GitHub Issues?
The issue-based system solves the HTTP 401 authentication problem that prevented automatic workflow triggering. Benefits include:
- ✅ No authentication errors
- ✅ Automatic processing
- ✅ Results delivered to you
- ✅ Email notifications
- ✅ Searchable history

### How long does processing take?
Typically 1-2 minutes, depending on:
- Query complexity
- MorphoSource API response time
- ChatGPT processing time

### Where do I see the results?
Results are posted in **two places**:
1. **Issue Comment** (recommended): Automatically posted to your issue
2. **GitHub Actions**: Available in the workflow run artifacts

### Can I ask follow-up questions?
Yes! You can:
- Comment on the closed issue to discuss results
- Create a new issue with a refined query
- Reference previous issue numbers in new queries

### What if something goes wrong?
If the workflow fails:
1. Check the workflow run logs in the Actions tab
2. Look for error messages in issue comments
3. Try rephrasing your query
4. Open a bug report issue (without the `query-request` label)

## 🔧 Technical Details

### System Architecture
```
User fills form
    ↓
Click creates GitHub Issue (with query-request label)
    ↓
Issue trigger workflow detects new issue
    ↓
Extracts query and triggers Query Processor
    ↓
Query Processor runs (MorphoSource API + ChatGPT)
    ↓
Results posted as comment + Issue closed
    ↓
User receives GitHub notification
```

### Required Permissions
The workflows need these permissions (already configured):
- `issues: write` - To create comments
- `actions: write` - To trigger workflows
- `contents: read` - To access code

### Rate Limits
- **GitHub API**: 5,000 requests/hour per user
- **MorphoSource API**: Depends on API key (typically generous)
- **OpenAI API**: Repository owner's quota

## 📝 Tips for Best Results

### Write Clear Queries
❌ Bad: "lizards"
✅ Good: "Tell me about lizard specimens with CT scan data"

### Be Specific
❌ Bad: "show me stuff"
✅ Good: "What Anolis specimens are available with micro-CT scans?"

### Use Natural Language
The system understands conversational queries:
- "How many..." 
- "Tell me about..."
- "Show me..."
- "Find specimens with..."

### Start Broad, Then Narrow
If your first query is too specific and returns no results:
1. Try a broader query
2. Examine the results
3. Refine with a follow-up query

## 🆘 Troubleshooting

### Issue doesn't trigger workflow
**Check:**
- Issue title starts with "Query:" or body contains "MorphoSource Query Submission"
- You're in the correct repository
- Workflows are enabled (check Settings → Actions)

**Solution:**
- Ensure your issue follows the query submission format (use the web form)
- The `query-request` label is added automatically when a query is detected
- Or manually trigger the workflow from Actions tab

### No results found
**Possible reasons:**
- No matching specimens in MorphoSource
- Query too specific
- API temporarily unavailable

**Solution:**
- Try a broader query
- Check [morphosource.org](https://www.morphosource.org) directly
- Wait a few minutes and try again

### Workflow fails
**Check:**
- Workflow run logs in Actions tab
- Issue comments for error messages

**Common causes:**
- OpenAI API key not configured (contact repo owner)
- MorphoSource API temporarily down
- Rate limit exceeded

### Results are slow
**Normal processing:**
- Simple query: 30-60 seconds
- Complex query: 1-2 minutes

**If longer:**
- Check workflow status in Actions tab
- Look for queued workflows (GitHub may be busy)

## 📚 Additional Resources

- [Complete Query System Guide](QUERY_SYSTEM_GUIDE.md)
- [Repository README](https://github.com/johntrue15/MorphoClaw/blob/main/README.md)
- [GitHub Issues Help](https://docs.github.com/en/issues)
- [MorphoSource](https://www.morphosource.org)

---

**Need help?** Open an issue (without the `query-request` label) or contact the repository maintainers.
