# Emergency Response Agent

**AI-Powered Emergency Triage System using NVIDIA NGC Fine-Tuned Llama + VAPI**

## Overview

This agent uses your fine-tuned Llama model (hosted on NVIDIA NGC) to analyze real-time health emergencies and automatically:

1. ✅ **Classify urgency** (0-10 scale)
2. ✅ **Find nearest facilities** (pharmacy, hospital, clinic)
3. ✅ **Determine response** (311 non-urgent vs 911 urgent)
4. ✅ **Make voice calls** via VAPI to appropriate services
5. ✅ **Provide immediate instructions** to caller

---

## 🚨 SAFETY NOTICE

**PLACEHOLDER MODE ENABLED**

This agent uses **PLACEHOLDER phone numbers** and will **NOT actually call 311 or 911**.

```python
PHONE_311_PLACEHOLDER = "+1-555-0311-000"  # NOT REAL
PHONE_911_PLACEHOLDER = "+1-555-0911-000"  # NOT REAL
```

To enable real emergency calls, you must:
1. Obtain proper authorization from emergency services
2. Update phone numbers in configuration
3. Test extensively in staging environment
4. Comply with all local emergency dispatch regulations

---

## Architecture

```
Real-Time Emergency Event
         ↓
[Emergency Response Agent]
         ↓
   1. Geospatial Analysis → Find nearest facilities
         ↓
   2. NVIDIA NGC Llama → Urgency classification
         ↓
   3. Decision Logic → 311 vs 911
         ↓
   4. VAPI Integration → Voice call (PLACEHOLDER)
         ↓
   Response Recommendation
```

---

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements_emergency.txt
```

### 2. Configure API Keys

```bash
# Copy example config
cp .env.example .env

# Edit .env and add your keys
nano .env
```

Required keys:
- `NGC_API_KEY`: Your NVIDIA NGC API key
- `NGC_MODEL_ENDPOINT`: URL to your fine-tuned Llama model
- `VAPI_API_KEY`: Your VAPI API key (optional for testing)

### 3. Load Facility Data (Optional)

The script includes sample SF facilities. To use full dataset:

```python
# In emergency_response_agent.py, line ~180
import json
with open('sf_health_facilities.json', 'r') as f:
    SF_FACILITIES = json.load(f)
```

---

## Usage

### Basic Demo

```bash
python emergency_response_agent.py
```

This runs two demos:
1. **Critical overdose emergency** → Should trigger 911 placeholder
2. **Non-urgent pharmacy need** → Should trigger 311 placeholder

### Custom Event

```python
from emergency_response_agent import EmergencyResponseAgent, EmergencyEvent, Location
from datetime import datetime

# Create emergency event
event = EmergencyEvent(
    event_id="TEST-001",
    timestamp=datetime.now(),
    location=Location(
        latitude=37.7749,
        longitude=-122.4194,
        address="123 Market St, San Francisco, CA"
    ),
    symptoms=["chest pain", "shortness of breath"],
    patient_age=55,
    patient_condition="conscious",
    witnessed_overdose=False,
    naloxone_available=False,
    caller_relation="self",
    additional_context="Patient experiencing chest pain for 30 minutes"
)

# Process emergency
agent = EmergencyResponseAgent()
recommendation = agent.process_emergency(event)

print(f"Urgency: {recommendation.urgency_score}/10")
print(f"Call: {recommendation.call_number}")
print(f"Action: {recommendation.recommended_action}")
```

---

## How It Works

### 1. **Geospatial Analysis**

```python
# Finds nearest facilities using Haversine formula
nearest_pharmacy = find_nearest_facilities(
    event.location, facilities_db, "Pharmacy"
)
nearest_hospital = find_nearest_facilities(
    event.location, facilities_db, "General Acute Care Hospital"
)
```

**Output:**
- Distance in miles
- Estimated travel time (assumes 25 mph average)
- Facility details (name, services, address)

---

### 2. **AI Urgency Classification**

The fine-tuned Llama model receives a detailed prompt:

```
Instruction: Analyze this real-time emergency situation...

Input:
**Emergency Event:** OD-2026-001
**Symptoms:** unresponsive, shallow breathing, blue lips, pinpoint pupils
**Patient Condition:** unconscious
**Witnessed Overdose:** Yes
**Naloxone Available:** Yes
**Nearby Resources:**
  Nearest hospital: UCSF Medical Center (1.2 miles, ~2.9 min)
  Nearest pharmacy: Walgreens #9342 (0.3 miles, ~0.7 min)

Output (JSON format):
{
  "urgency_score": 9.5,
  "urgency_level": "CRITICAL",
  "call_number": "911",
  "recommended_action": "IMMEDIATE 911 - Suspected opioid overdose",
  "immediate_instructions": [
    "Call 911 immediately",
    "Administer naloxone now (2mg nasal spray)",
    "Place patient in recovery position",
    "Monitor breathing - perform rescue breathing if needed",
    "Stay on scene until EMS arrives"
  ],
  "rationale": "Patient exhibits classic opioid overdose symptoms: unconscious,
                respiratory depression, pinpoint pupils. Naloxone available on scene.
                Immediate administration can save life. EMS response critical."
}
```

**Fine-tuned model advantages:**
- Trained on 897 SF health records
- Understands local EMS response times (3.67 min avg)
- Knows naloxone distribution patterns
- Familiar with SF facility network

---

### 3. **Decision Logic**

```python
if urgency_score >= 7.0:
    call_number = PHONE_911_PLACEHOLDER  # Critical → 911
else:
    call_number = PHONE_311_PLACEHOLDER  # Non-urgent → 311
```

**Urgency Levels:**
- **9-10 CRITICAL**: Life-threatening (overdose, cardiac arrest, severe trauma)
- **7-8 HIGH**: Urgent medical need (chest pain, severe bleeding)
- **5-6 MODERATE**: Non-life-threatening but needs care (fracture, infection)
- **0-4 LOW**: Minor issues (prescription refill, health advice)

---

### 4. **VAPI Voice Call**

```python
vapi_client.make_call(
    phone_number="+1-555-0911-000",  # PLACEHOLDER
    message="Emergency dispatch, this is an automated report for event OD-2026-001.
             Critical situation at 455 Golden Gate Ave. Patient is unconscious,
             symptoms include unresponsive, shallow breathing, blue lips.
             Suspected drug overdose. Naloxone is available on scene.
             Nearest hospital is UCSF Medical Center, 1.2 miles away.
             Immediate medical response required.",
    urgency_level="CRITICAL",
    event_id="OD-2026-001"
)
```

**VAPI Features:**
- Realistic AI voice (11Labs)
- Adjustable speaking rate (faster for emergencies)
- Recording enabled for audit trail
- Metadata tracking (event ID, timestamp, urgency)

---

## Example Outputs

### Demo 1: Critical Overdose

```
======================================================================
🚨 EMERGENCY EVENT: OD-2026-001
======================================================================
Time: 2026-01-25 12:45:32
Location: 37.7838, -122.4167
Symptoms: unresponsive, shallow breathing, blue lips, pinpoint pupils
Condition: unconscious
Overdose witnessed: True
Naloxone available: True

📍 Finding nearest facilities...
   Pharmacy: Walgreens Pharmacy #9342 (0.31 mi)
   Hospital: California Pacific Med Ctr-pacific Campus (0.54 mi)
   Clinic: Baart Turk Street Clinic (0.41 mi)

🤖 Consulting fine-tuned Llama model...

======================================================================
📋 RECOMMENDATION
======================================================================
Urgency Score: 9.5/10 (CRITICAL)
Call Number: +1-555-0911-000
Action: IMMEDIATE 911 - Suspected opioid overdose

🔴 IMMEDIATE INSTRUCTIONS:
   1. Call 911 immediately
   2. Administer naloxone now (2mg nasal spray)
   3. Place patient in recovery position
   4. Monitor breathing - perform rescue breathing if needed
   5. Stay on scene until EMS arrives

📍 NEAREST FACILITIES:
   Hospital: California Pacific Med Ctr-pacific Campus (0.54 mi)
   Clinic: Baart Turk Street Clinic (0.41 mi)
   Pharmacy: Walgreens Pharmacy #9342 (0.31 mi)

⏱️  Estimated EMS arrival: 3.7 minutes

💡 Rationale: Patient exhibits classic opioid overdose symptoms with
   respiratory depression. Naloxone available on scene - immediate
   administration critical. EMS dispatched to scene.

📞 SIMULATED CALL to +1-555-0911-000
   Message: Emergency dispatch, this is an automated report...
   [PLACEHOLDER - NOT ACTUALLY CALLING]

📞 Call Status: initiated
   Call ID: SIMULATED-OD-2026-001
```

---

### Demo 2: Non-Urgent Pharmacy

```
======================================================================
🚨 EMERGENCY EVENT: PHARM-2026-001
======================================================================
Time: 2026-01-25 12:46:15
Location: 37.7749, -122.4194
Symptoms: withdrawal symptoms, anxiety, nausea
Condition: conscious
Overdose witnessed: False
Naloxone available: False

📍 Finding nearest facilities...
   Pharmacy: CVS Pharmacy #9871 (0.18 mi)
   Hospital: UCSF Medical Center (1.42 mi)
   Clinic: Baart Turk Street Clinic (0.23 mi)

🤖 Consulting fine-tuned Llama model...

======================================================================
📋 RECOMMENDATION
======================================================================
Urgency Score: 5.5/10 (MODERATE)
Call Number: +1-555-0311-000
Action: Contact 311 for health resources - Assist with prescription refill

🔴 IMMEDIATE INSTRUCTIONS:
   1. Visit nearest pharmacy: CVS Pharmacy #9871 (0.18 miles)
   2. Contact prescribing physician for refill authorization
   3. If symptoms worsen, call 911
   4. Consider visiting nearest clinic if pharmacy closed

📍 NEAREST FACILITIES:
   Hospital: UCSF Medical Center (1.42 mi)
   Clinic: Baart Turk Street Clinic (0.23 mi)
   Pharmacy: CVS Pharmacy #9871 (0.18 mi)

⏱️  Estimated EMS arrival: 3.7 minutes

💡 Rationale: Patient experiencing mild withdrawal symptoms due to
   medication gap. Not immediately life-threatening but requires
   prompt medication access. Nearest pharmacy can assist.

📞 SIMULATED CALL to +1-555-0311-000
   Message: San Francisco 311, this is an automated health advisory...
   [PLACEHOLDER - NOT ACTUALLY CALLING]

📞 Call Status: initiated
   Call ID: SIMULATED-PHARM-2026-001
```

---

## Customization

### Adjust Urgency Threshold

```python
# In emergency_response_agent.py or .env
URGENCY_SCORE_THRESHOLD = 7.0  # Default: >= 7 triggers 911

# Lower threshold (more calls to 911):
URGENCY_SCORE_THRESHOLD = 6.0

# Higher threshold (fewer calls to 911):
URGENCY_SCORE_THRESHOLD = 8.0
```

### Add Custom Facilities

```python
SF_FACILITIES.append({
    "facility_name": "Your Custom Clinic",
    "facility_type": "Community Clinic",
    "services": "Primary Care",
    "location": {
        "latitude": "37.7500",
        "longitude": "-122.4000",
        "human_address": '{"address": "123 Main St", "city": "San Francisco"}'
    }
})
```

### Modify Model Parameters

```python
# Adjust creativity vs consistency
llama_client.generate_response(
    prompt,
    max_tokens=1024,
    temperature=0.5  # Lower = more consistent, Higher = more creative
)
```

---

## API Endpoints

### NVIDIA NGC

Your fine-tuned model endpoint:
```
POST https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/{YOUR-FUNCTION-ID}

Headers:
  Authorization: Bearer {NGC_API_KEY}
  Content-Type: application/json

Body:
{
  "messages": [{"role": "user", "content": "..."}],
  "temperature": 0.7,
  "max_tokens": 1024
}
```

### VAPI

Voice call endpoint:
```
POST https://api.vapi.ai/call

Headers:
  Authorization: Bearer {VAPI_API_KEY}
  Content-Type: application/json

Body:
{
  "phoneNumberId": "...",
  "customer": {"number": "+1-555-0911-000"},
  "assistant": {...}
}
```

---

## Testing

### Unit Tests (Recommended)

```python
def test_urgency_classification():
    """Test model classifies overdose as critical"""
    event = create_overdose_event()
    agent = EmergencyResponseAgent()
    rec = agent.process_emergency(event)

    assert rec.urgency_score >= 8.0
    assert rec.urgency_level == "CRITICAL"
    assert "911" in rec.call_number

def test_pharmacy_routing():
    """Test non-urgent pharmacy need routes to 311"""
    event = create_pharmacy_event()
    agent = EmergencyResponseAgent()
    rec = agent.process_emergency(event)

    assert rec.urgency_score < 7.0
    assert "311" in rec.call_number
```

### Integration Testing

```bash
# Test with real NGC endpoint (but placeholder calls)
NGC_API_KEY=your-real-key python emergency_response_agent.py

# Test with mock NGC endpoint (for CI/CD)
NGC_MODEL_ENDPOINT=http://localhost:8000/mock python emergency_response_agent.py
```

---

## Production Deployment

### Requirements

1. **Legal/Regulatory:**
   - Authorization from emergency services (911/311)
   - HIPAA compliance for patient data
   - Recording consent notifications
   - Data retention policies

2. **Technical:**
   - High-availability infrastructure (99.9% uptime)
   - Redundant API endpoints
   - Call queue management
   - Real-time monitoring & alerting

3. **Testing:**
   - 1000+ test cases across urgency levels
   - Load testing (100+ concurrent events)
   - Failover testing
   - False positive/negative analysis

### Deployment Checklist

- [ ] NGC model fine-tuned and validated
- [ ] VAPI account configured with real phone numbers
- [ ] Emergency services coordination completed
- [ ] Update `PHONE_311_PLACEHOLDER` → real 311 number
- [ ] Update `PHONE_911_PLACEHOLDER` → real 911 number
- [ ] Implement logging & monitoring (DataDog, Splunk)
- [ ] Set up incident response procedures
- [ ] Train human dispatchers on AI-assisted workflow
- [ ] Establish escalation protocols (AI → human)
- [ ] Regular model retraining (quarterly with new SF data)

---

## Troubleshooting

### Model Returns Generic Response

**Issue:** Fine-tuned model not producing detailed JSON output

**Fix:**
```python
# Add more explicit JSON formatting to prompt
prompt += "\n\nIMPORTANT: Return ONLY valid JSON with these exact fields:..."
```

### VAPI Calls Failing

**Issue:** VAPI returns 401 Unauthorized

**Fix:**
```bash
# Check API key
echo $VAPI_API_KEY

# Regenerate key at vapi.ai dashboard
```

### Wrong Urgency Classification

**Issue:** Model classifies critical events as low urgency

**Fix:**
1. Review training data for similar cases
2. Add more critical examples to training set
3. Retrain model with additional data
4. Lower `URGENCY_SCORE_THRESHOLD` temporarily

---

## Limitations

1. **Geographic Scope:** Trained on SF data only (generalization may vary)
2. **Language:** English only (multilingual requires retraining)
3. **Connectivity:** Requires internet for NGC API calls
4. **Latency:** ~2-5 seconds from event to recommendation
5. **Accuracy:** Not 100% - human oversight recommended

---

## License

MIT License (code)
Public Domain (SF Open Data)

---

## Support

**Technical Issues:** Create issue in GitHub repo
**Emergency Services Integration:** Contact your local dispatch coordinator
**NVIDIA NGC:** https://ngc.nvidia.com/support
**VAPI:** https://docs.vapi.ai

---

## Changelog

**v1.0.0** (2026-01-25)
- Initial release
- NVIDIA NGC Llama integration
- VAPI voice calling
- SF health facilities database
- Placeholder phone numbers (safe mode)
