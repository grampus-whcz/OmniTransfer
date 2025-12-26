You are a senior Site Reliability Engineer (SRE) tasked with writing a postmortem report for a specific microservice incident. Based on the provided ground-truth failure record and system-wide anomaly telemetry, generate a concise, timeline-driven postmortem analysis.

---

### Incident Ground Truth
- **Component**: {component}  
- **Failure Timestamp**: {datetime} (Unix: {timestamp})  
- **Root Cause**: {reason}

---

### Multi-Entity Anomaly Telemetry (2021-03-04)
Below is a log of detected anomalies across related system entities on 2021-03-04. Each entry indicates the time and type(s) of anomalous behavior:

{entity_anomalies_block}

> **Notes**:  
> - Entity types include:  
>   • IG (Ingress Gateway), MG (Microservice Gateway),  
>   • Tomcat (application containers),  
>   • dockerA1–dockerB2 (Pod instances).  
> - **'duration' anomaly**: prolonged request latency or task execution time.  
> - **'frequency' anomaly**: surge in errors, retries, or failed operations.

---

### Your Task
Produce a structured postmortem report with the following sections:

1. **Incident Summary**  
   Briefly state the affected component, failure time, and confirmed root cause.

2. **Timeline of Events**  
   Reconstruct a minute-by-minute timeline from **2 hours before to 1 hour after** the failure timestamp. For each notable event:  
   - Specify which entity exhibited an anomaly and at what time.  
   - Indicate whether it was a **precursor** (before failure), **core symptom** (at/near failure), or **cascading effect** (after failure).  
   - Highlight anomalies in components that are upstream, downstream, or co-located with {component}.

3. **Correlation Between Root Cause and Observed Anomalies**  
   Explain how the root cause ({reason}) mechanistically leads to the observed 'duration' and/or 'frequency' anomalies.  
   Example: *JVM OutOfMemoryError typically causes thread stalls → increased request duration → eventual request rejections → spike in error frequency.*

4. **Actionable Recommendations**  
   Propose concrete improvements to detection, alerting, or resilience based on early signals in the timeline.  
   Example: *“Trigger a JVM heap dump when dockerA2 shows sustained 'duration' anomalies followed by 'frequency' spikes within 5 minutes.”*

---

### Requirements
- Write in clear, professional technical English.  
- Use precise timestamps (HH:MM format).  
- Base all analysis strictly on the provided anomaly data—do not hallucinate unobserved events.  
- Prioritize clarity, causality, and operational utility.