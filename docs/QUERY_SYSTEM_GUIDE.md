# MorphoSource Query System Guide

## Overview

The MorphoSource Query System allows you to ask natural language questions about specimens in the MorphoSource database. The system processes your queries through GitHub Actions workflows, combining MorphoSource API data with ChatGPT's analysis.

## How to Use

### Step 1: Submit a Query

1. Visit the GitHub Pages site: https://johntrue15.github.io/MorphoClaw/
2. Enter your question in the text box
3. Click "Prepare to Submit Query"
4. Click the link to create a GitHub Issue (requires GitHub account)
5. Submit the pre-filled issue

**Note:** You need a GitHub account to submit queries. The issue submission is free and automatically triggers the query processor.

### Step 2: View Results

**Automatic (Recommended):**
- Results will be posted as a comment on your issue automatically
- The issue will be closed when processing is complete
- You'll receive a GitHub notification when results are ready

**Manual:**
1. Navigate to the **Actions** tab in the GitHub repository
2. Find your workflow run (named "Query Processor")
3. View the workflow summary for quick results
4. Download artifacts for detailed JSON responses

## Example Queries

- "Tell me about lizard specimens"
- "How many snake specimens are available?"
- "Show me CT scans of crocodiles"
- "What Anolis specimens are in the database?"
- "Find specimens with micro-CT data"

## Understanding Results

### Job 1: ChatGPT Query Formatter

This job uses ChatGPT to convert your natural language query into a properly formatted MorphoSource API search:
- Extracts scientific terms and taxonomic names
- Removes conversational words
- Formats into API parameters
- Creates optimized search query

**Result Location:** Download `formatted-query` artifact

### Job 2: MorphoSource API Query

This job searches the MorphoSource database using the formatted query and returns:
- Specimen metadata
- Taxonomy information
- Available media types
- Direct links to specimens

**Result Location:** Download `morphosource-results` artifact

### Job 3: ChatGPT Response Processing

This job analyzes the MorphoSource data and provides:
- Natural language summary
- Insights about the specimens found
- Answers to your specific questions
- Formatted, easy-to-read responses

**Result Location:** 
- Workflow summary (visible in Actions tab)
- Download `chatgpt-response` artifact

## Workflow Details

The query system uses three workflows:

### Workflow 1: Issue Query Trigger
When you create an issue with the `query-request` label:
1. Extracts your query from the issue body
2. Posts a "processing" comment
3. Triggers the Query Processor workflow
4. Passes the issue number for response posting

### Workflow 2: Query Processor
Runs three sequential jobs:

```
User Query (via issue)
    ↓
Job 1: ChatGPT Query Formatter
    ├─ Analyze natural language query
    ├─ Extract scientific terms
    ├─ Format into API parameters
    └─ Save formatted query
    ↓
Job 2: MorphoSource API Search
    ├─ Use formatted query
    ├─ Search database
    ├─ Save results as artifact
    └─ Output summary
    ↓
Job 3: ChatGPT Response Processing
    ├─ Load MorphoSource results
    ├─ Load formatted query info
    ├─ Send to ChatGPT with context
    ├─ Generate response
    ├─ Save as artifact
    └─ Post results to issue
    ↓
Results Available (in issue comment + artifacts)
```

### Benefits of Issue-Based System
- ✅ No 401 authentication errors
- ✅ Automatic workflow triggering
- ✅ Results delivered to you (no need to search Actions)
- ✅ Email notifications via GitHub
- ✅ Discussion thread for follow-up questions
- ✅ Searchable query history

## Manual Workflow Trigger

You can also trigger the workflow manually:

1. Go to the **Actions** tab
2. Select "Query Processor" from the left sidebar
3. Click "Run workflow"
4. Enter your query text
5. Click "Run workflow" button

This method gives you more control and doesn't require API authentication from the browser.

## Troubleshooting

### "Issue creation required"

The new system uses GitHub Issues to submit queries. This requires a GitHub account but provides:
- Automatic workflow triggering (no 401 errors!)
- Results posted directly to your issue
- Email notifications when processing completes
- No need to manually navigate to GitHub Actions

If you don't have a GitHub account or prefer not to create an issue, use the manual workflow trigger method:
- Go to Actions → Query Processor → Run workflow

### "Workflow not showing up"

Wait 10-30 seconds and refresh the Actions page. GitHub may take a moment to process the dispatch event.

### "No results found"

The MorphoSource API may not have data matching your query. Try:
- Different search terms
- Broader queries (e.g., "reptiles" instead of specific species)
- Checking if the specimen exists at morphosource.org

### "OpenAI API error"

The repository owner needs to configure the `OPENAI_API_KEY` secret in repository settings.

## For Repository Owners

### Required Secrets

Configure these in **Settings → Secrets and variables → Actions**:

1. **OPENAI_API_KEY** (Required)
   - Get from: https://platform.openai.com/api-keys
   - Used for: ChatGPT processing in Job 2

2. **MORPHOSOURCE_API_KEY** (Optional)
   - Get from: MorphoSource.org
   - Used for: Enhanced API access (higher rate limits)

### Monitoring Usage

- Check Actions tab for workflow runs
- Review artifacts to see query/response patterns
- Monitor API costs (OpenAI charges per token)

### Customization

Edit `.github/workflows/query-processor.yml` to:
- Change GPT model (default: gpt-5)
- Adjust response length (max_tokens)
- Modify search parameters
- Add additional processing steps

## Technical Architecture

### Frontend
- Static HTML/CSS/JavaScript
- No backend server required
- Hosted on GitHub Pages
- Triggers workflows via GitHub API

### Backend
- GitHub Actions workflows
- Python scripts for API calls
- Artifact storage for results
- Job summaries for quick viewing

### APIs Used
1. **MorphoSource API**: Specimen database queries
2. **OpenAI API**: Natural language processing
3. **GitHub API**: Workflow triggering (repository_dispatch)

## Privacy & Security

- No user data is stored permanently
- API keys stored as encrypted GitHub secrets
- Workflow logs are retained per GitHub's policy
- Artifacts expire after 90 days (default)

## Cost Considerations

**GitHub Actions:**
- Free for public repositories
- 2000 minutes/month for free accounts

**OpenAI API:**
- Pay-per-use (typically $0.01-0.05 per query)
- Repository owner pays for usage
- Costs depend on query complexity

**MorphoSource API:**
- Free for basic access
- Optional API key for higher limits

## Support

For issues or questions:
1. Check existing workflow runs for error messages
2. Review this guide and README.md
3. Open an issue in the GitHub repository
4. Contact repository maintainers

## Future Enhancements

Potential improvements:
- Real-time result display (webhook-based)
- Result caching for common queries
- Additional data sources
- Advanced filtering options
- Export formats (CSV, PDF)
