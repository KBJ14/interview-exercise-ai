# RAG Checks

Manual check questions for the fictional Quillbox corpus in `data/documents/`.
Each answerable question lists the expected answer and the section that grounds it.
The unanswerable question checks that the assistant says the documents do not contain the answer instead of guessing.

| # | Question | Expected answer | Source (file / section) |
|---|----------|-----------------|-------------------------|
| 1 | How much storage does the Free plan include, and what happens when I reach it? | 1 GB. New uploads are blocked, but existing files stay viewable, editable, and downloadable. | `plans.txt` / Free Plan |
| 2 | What is the daily API request limit on the Team plan? | 50,000 requests per day per workspace. Extra requests get HTTP 429 until the limit resets at 00:00 UTC. | `plans.txt` / Team Plan; What Happens When You Exceed a Limit |
| 3 | Can I get a refund for a monthly renewal charge? | No. Monthly renewals are not refundable. Only first-time paid subscriptions and annual renewals get a full refund within 14 days. | `billing.txt` / Refund Policy |
| 4 | If I cancel my Pro subscription, when do I lose Pro features? | At the end of the current billing period. The workspace then moves to Free, and the account and data are kept. | `billing.txt` / Canceling a Subscription |
| 5 | How long is a password reset link valid? | 60 minutes, and each link can be used only once. | `account.txt` / Resetting Your Password |
| 6 | Can I recover my account after requesting deletion? | Yes, within the 30-day grace period, by signing in and selecting "Restore account". The account returns on the Free plan. After 30 days it cannot be recovered. | `account.txt` / Deletion Grace Period |
| 7 | Does Quillbox offer a student or nonprofit discount? | **No grounding in the documents.** None of the three documents mentions discounts. The expected behaviour is to say the information is not available, not to invent a policy. | None |
