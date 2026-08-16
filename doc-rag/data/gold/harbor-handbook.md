# Harbor Labs Employee Handbook (FY 2025–26)

Version 4.2 · Effective 1 April 2025 · Applies to all full-time, intern, and contractor staff in India unless a signed offer letter says otherwise.

## 1. Purpose and scope

This handbook is the source of truth for people operations at Harbor Labs. Managers may not invent local exceptions. If an offer letter conflicts with this handbook, the offer letter wins for that employee only.

Harbor Labs builds industrial inspection software. We operate from Bengaluru (HQ), Pune, and a small support desk in Gurgaon. Remote work is allowed up to three days per week after probation, with manager approval in Workday.

## 2. Working hours and attendance

Core hours are 11:00–16:00 IST, Monday to Friday. Teams may choose a 9:00–18:00 or 10:00–19:00 envelope around core hours. Contractors on night-shift support follow the roster published by Site Reliability.

Attendance is marked in Workday. More than four unexplained late marks in a calendar month triggers a written warning. Three written warnings in twelve months can lead to a performance improvement plan.

Public holidays follow the Harbor Labs India holiday calendar. Floating holidays: two per year, requested at least five working days in advance.

## 3. Leave

Annual leave (privilege leave) is 18 days in year one and 21 days from year two. Leave accrues monthly and can be carried forward up to 8 days. Encashment is not allowed except at exit.

Sick leave is 12 days per year. A medical certificate is required for absences longer than two consecutive working days.

Casual leave is 6 days per year and cannot be combined with privilege leave to extend a long weekend beyond five calendar days without VP approval.

Maternity leave is 26 weeks as required by the Maternity Benefit Act. **Paternity leave is 15 working days**, to be taken within 120 days of the child's birth or adoption. Unused paternity leave lapses and is not encashed.

Bereavement leave is 5 working days for immediate family (spouse, child, parent, sibling).

Sabbatical: employees with 5+ years of continuous service may request 8 weeks unpaid sabbatical. Approval sits with the CHRO.

## 4. Notice period and exit

Individual contributors have a **60-day notice period**. People managers (anyone with direct reports in Workday) have a **90-day notice period**. Harbor Labs may allow buy-out of remaining notice at 100% of gross monthly pay, at company discretion.

On the last working day, IT collects the laptop, access badge, and hardware token. Final settlement is paid within 45 days of exit, after clearance from IT, Finance, and the reporting manager.

Non-solicit of Harbor Labs customers lasts 12 months after exit. Non-solicit of employees lasts 6 months.

## 5. Compensation, internships, and expenses

Salaries are paid on the last working day of each month. Tax documents are in Darwinbox.

Interns (minimum 8 weeks) receive a **stipend of INR 40,000 per month**, credited on the same payroll cycle. Interns are not eligible for privilege leave or variable pay.

Travel and expenses: book flights on the Harbor Labs corporate portal. Personal cards may be used only when the portal is down. **Original GST invoices are required for any expense above INR 750.** Below INR 750, a photo of the bill plus a one-line note in Expensify is enough. Alcohol is never reimbursable. Tips above 10% need manager comment.

Local conveyance in Bengaluru uses the company Uber for Business profile. Auto-rickshaw claims need a Fastag/Paytm screenshot.

## 6. IT, devices, and security

Every full-time employee receives a company laptop. **Disk encryption is mandatory: BitLocker on Windows and FileVault on macOS.** IT will not issue admin rights until encryption is confirmed in Jamf/Intune.

VPN: use Harbor Gateway. **MFA push codes expire in 8 hours.** Sharing a VPN session is a P1 security incident.

Guest network SSID is **Harbor-Guest**. The passphrase is rotated every Monday at 09:00 IST and posted on the #it-announcements Slack channel. Production SSH keys never go on guest Wi-Fi.

USB storage is blocked by policy. Exceptions require a ticket to IT Security with a ticket SLA of two business days.

Lost laptop: lock and wipe via MDM within 2 hours of the report. File an incident in Jira Service Management under Security.

## 7. Helpdesk SLAs

Severity definitions:

- P1 (production down or suspected data leak): first response in **30 minutes**, 24×7.
- P2 (major feature broken, workaround exists): first response in 4 business hours.
- P3 (how-to and access): first response in 1 business day.

The helpdesk email is it-help@harborlabs.example. Do not CC personal Gmail.

## 8. Data and vendors

Customer inspection images are stored in ap-south-1. **Access logs are retained for 18 months**, then deleted. Engineers must not copy customer images to laptops.

Background verification for new hires is run by **AuthBridge**. Offers are contingent on a clear report within 15 calendar days.

The only approved LLM tools for source code are the Harbor private Azure OpenAI deployment. Public ChatGPT and Gemini are not approved for customer data or unreleased product specs.

## 9. Conduct

Harbor Labs has a zero-tolerance policy for harassment. Reports go to people-ops@harborlabs.example or the anonymous form on the intranet. Retaliation against a reporter is itself a fireable offense.

Gifts from vendors above INR 2,000 must be declared to Finance. Cash gifts are never allowed.

## 10. Document control

This handbook is reviewed every April. Questions: people-ops@harborlabs.example. The PDF in Workday is the signed copy; printed posters on walls are not authoritative.

## 11. Nearby facts that are easy to mix up

These lines exist so retrieval cannot cheat on a short number match:

- The HQ cafeteria lunch rush lasts about 30 minutes after 13:00.
- Campus joining bonus for some intern-to-FTE conversions is INR 40,000 one-time, not a stipend.
- Printer toner purchase orders use a 60-day vendor credit cycle. That is not an employee notice period.
- Building generators are tested for 8 hours every quarter. That is not the VPN MFA window.
- The facilities MSA with CleanPro runs 18 months. That is not log retention.
- Auth0 is used for the customer product login. Auth0 is not the background-check vendor.
- FileVault-only is not enough on Windows. Windows must use BitLocker as stated in section 6.

