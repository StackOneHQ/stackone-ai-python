# Search Benchmark Results

## Local BM25+TF-IDF vs Semantic Search

**Date:** 2025-02-06
**Dataset:** 94 evaluation tasks across 8 categories
**Corpus:** 5,144 actions from 200+ connectors
**Metric:** Hit@5 (correct action in top 5 results)

## Summary

| Method            | Hit@5      | MRR        | Avg Latency | Hits    |
| ----------------- | ---------- | ---------- | ----------- | ------- |
| Local BM25+TF-IDF | 66.0%      | 0.538      | 1.2ms       | 62/94   |
| Semantic Search   | 76.6%      | 0.634      | 279.6ms     | 72/94   |
| **Improvement**   | **+10.6%** | **+0.096** |             | **+10** |

## Detailed Breakdown

### Semantic Wins (17 tasks)

Tasks where semantic search finds the correct result but local BM25 fails.
These demonstrate semantic search's ability to understand **intent and synonyms**.

| Query                             | Local Top Result                | Semantic Top Result              |
| --------------------------------- | ------------------------------- | -------------------------------- |
| "fire someone"                    | workable_get_job_recruiters     | factorial_terminate_employee     |
| "ping the team"                   | teamtailor_delete_team          | slack_send_message               |
| "file a new bug"                  | github_create_or_update_file    | jira_update_issue                |
| "ping my colleague"               | salesforce_get_my_events        | microsoftoutlook_reply_message   |
| "fetch staff information"         | pinpoint_get_application        | workday_list_workers             |
| "show me everyone in the company" | humaans_get_me                  | lattice_talent_list_users        |
| "turn down a job seeker"          | pinpoint_get_job_seeker         | jobadder_reject_requisition      |
| "check application status"        | dropbox_check_remove_member     | jobadder_list_application_status |
| "check my to-do list"             | jira_check_bulk_permissions     | todoist_list_tasks               |
| "start a group chat"              | microsoftteams_update_chat      | discord_create_group_dm          |
| "move candidate forward"          | workable_move_candidate         | greenhouse_move_application      |
| "approve PTO"                     | ashby_approve_offer             | planday_approve_absence_request  |
| "update staff record"             | bamboohr_update_hour_record     | cezannehr_update_employee        |
| "pull the org chart"              | github_create_issue_comment     | lattice_list_review_cycles       |
| "assign training to employee"     | easyllama_assign_training       | hibob_create_training_record     |
| "file a bug report"               | smartrecruiters_get_report_file | github_create_issue_comment      |
| "track customer interaction"      | qlik_create_interaction         | peoplefluent_track_launch        |

### Local Wins (7 tasks)

Tasks where BM25 keyword matching outperforms semantic search.

| Query                               | Local Top Result                       | Semantic Top Result           |
| ----------------------------------- | -------------------------------------- | ----------------------------- |
| "see who applied for the role"      | greenhouse_list_applied_candidate_tags | ashby_add_hiring_team_member  |
| "advance someone to the next round" | greenhouse_move_application            | factorial_invite_employee     |
| "see open positions"                | teamtailor_list_jobs                   | hibob_create_position_opening |
| "close a deal"                      | zohocrm_get_deal                       | shopify_close_order           |
| "check course completion"           | saba_delete_recurring_completion       | saba_get_course               |
| "update deal and notify team"       | zohocrm_get_deal                       | microsoftteams_update_team    |
| "look up customer"                  | linear_update_customer_need            | shopify_search_customers      |

### Both Miss (15 tasks)

Hard queries that neither method handles well. Many are abbreviations, cross-domain concepts, or have overly strict expected matches.

| Query                       | Category  | Why Hard                                                       |
| --------------------------- | --------- | -------------------------------------------------------------- |
| "onboard a new team member" | hr        | "team member" maps to team tools, not HR                       |
| "OOO"                       | hr        | Abbreviation - neither understands                             |
| "DM someone"                | messaging | Both find discord_create_dm but expected pattern too strict    |
| "customer onboarding"       | crm       | Cross-domain concept                                           |
| "close quarter books"       | crm       | Domain-specific financial term                                 |
| "PTO request"               | hr        | Both find PTO tools but expected pattern mismatch              |
| "kill the ticket"           | project   | Both find delete_ticket but expected pattern mismatch          |
| "who works in engineering"  | hr        | Requires department filtering, not just listing                |
| "add a new prospect"        | crm       | Both find prospect tools but connector mismatch                |
| "see all shared files"      | documents | "shared" narrows scope too much                                |
| "see available trainings"   | lms       | Both find training tools but pattern mismatch                  |
| "track learning progress"   | lms       | Abstract concept mapping                                       |
| "create team workspace"     | messaging | Cross-domain: workspace vs channel                             |
| "log customer call"         | crm       | Connector-specific (Salesforce) term                           |
| "add new lead"              | crm       | Connector-specific (HubSpot) but returns wrong HubSpot actions |

## How to Run

### Local Mode (recommended for development)

Requires the action_search Lambda running locally:

```bash
# Terminal 1: Start the Lambda
cd ai-generation/apps/action_search
cp .env.example .env
# Edit .env: set USE_LOCAL_STORE=false and TURBOPUFFER_API_KEY=tpuf_xxx
make run-local

# Terminal 2: Run benchmark
cd stackone-ai-python
uv run python tests/benchmark_search.py --local
```

### Production Mode

```bash
STACKONE_API_KEY=xxx uv run python tests/benchmark_search.py
```

### CLI Options

```
--local              Use local Lambda instead of production API
--lambda-url URL     Custom Lambda URL (default: localhost:4513)
--api-url URL        Custom production API URL
```

## Methodology

### Evaluation Tasks

94 tasks across 8 categories:

| Category           | Tasks | Description                                  |
| ------------------ | ----- | -------------------------------------------- |
| HR/HRIS            | 19    | Employee management, time off, org structure |
| Recruiting/ATS     | 12    | Candidates, applications, interviews         |
| CRM                | 12    | Contacts, deals, accounts                    |
| Project Management | 8     | Tasks, issues, projects                      |
| Messaging          | 5     | Messages, channels, conversations            |
| Documents          | 5     | Files, folders, drives                       |
| Marketing          | 5     | Campaigns, lists, automation                 |
| LMS                | 5     | Courses, assignments, completions            |

Plus per-connector tests (Slack, Jira, Greenhouse, Salesforce, HubSpot) and edge cases (abbreviations, slang, complex queries).

### Matching Logic

- **Hit@5**: At least one expected pattern appears (case-insensitive partial match) in the top 5 results
- **MRR** (Mean Reciprocal Rank): 1/position of first correct result, averaged across all tasks
- **Fair comparison**: Both methods search the same 5,144-action corpus

### Corpus

Both local and semantic search operate on the same action catalog:

- 5,144 unique actions
- 200+ connectors (BambooHR, Greenhouse, Salesforce, Slack, Jira, etc.)
- 7 verticals (HRIS, ATS, CRM, Documents, IAM, LMS, Marketing)

## Conclusions

1. **Semantic search improves accuracy by +10.6%** (66.0% -> 76.6% Hit@5)
2. **Semantic excels at intent understanding**: "fire someone" -> terminate, "ping the team" -> send_message
3. **Local BM25 is competitive** when queries contain exact keywords from tool names
4. **15 tasks need better evaluation criteria** - some "misses" are actually correct results with overly strict expected patterns
5. **Latency tradeoff**: Local is ~230x faster (1.2ms vs 280ms) but runs in-memory with pre-built index
