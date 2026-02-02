#!/usr/bin/env python3
"""
Emergency Response Agent - NVIDIA NGC Llama Integration with VAPI

This script uses the fine-tuned Llama model hosted on NVIDIA NGC to analyze
real-time health emergencies and determine optimal response actions including:
- Nearest pharmacy, hospital/clinic, first responders
- Urgency classification (311 non-urgent vs 911 urgent)
- Automated voice calls via VAPI

SAFETY: Uses PLACEHOLDER phone numbers - does NOT actually call 311/911
"""

import json
import os
import math
import requests
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ============================================================================
# CONFIGURATION
# ============================================================================

# NVIDIA NGC Configuration
NGC_API_KEY = os.getenv("NGC_API_KEY", "nvapi-PLACEHOLDER-KEY")
NGC_MODEL_ENDPOINT = os.getenv("NGC_MODEL_ENDPOINT",
    "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/llama-healthcare-optimized")

# VAPI Configuration (Voice API)
VAPI_API_KEY = os.getenv("VAPI_API_KEY", "vapi-PLACEHOLDER-KEY")
VAPI_ENDPOINT = "https://api.vapi.ai/call"

# PLACEHOLDER NUMBERS - NOT REAL 311/911
PHONE_311_PLACEHOLDER = "+1-555-0311-000"  # Non-urgent services
PHONE_911_PLACEHOLDER = "+1-555-0911-000"  # Emergency services

# Emergency thresholds
URGENCY_SCORE_THRESHOLD = 7.0  # Score >= 7 triggers 911, < 7 triggers 311


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Location:
    """Geographic location"""
    latitude: float
    longitude: float
    address: Optional[str] = None


@dataclass
class EmergencyEvent:
    """Real-time emergency event data"""
    event_id: str
    timestamp: datetime
    location: Location
    symptoms: List[str]
    patient_age: Optional[int]
    patient_condition: str  # "conscious", "unconscious", "altered"
    witnessed_overdose: bool
    naloxone_available: bool
    caller_relation: str  # "self", "bystander", "family"
    additional_context: str


@dataclass
class Facility:
    """Healthcare facility"""
    name: str
    facility_type: str
    services: str
    location: Location
    distance_miles: float
    estimated_response_time_min: Optional[float] = None


@dataclass
class ResponseRecommendation:
    """Model's recommended response"""
    urgency_score: float  # 0-10 scale
    urgency_level: str  # "CRITICAL", "HIGH", "MODERATE", "LOW"
    call_number: str  # 911 or 311 placeholder
    recommended_action: str
    nearest_pharmacy: Optional[Facility]
    nearest_hospital: Optional[Facility]
    nearest_clinic: Optional[Facility]
    estimated_ems_arrival_min: Optional[float]
    immediate_instructions: List[str]
    rationale: str


# ============================================================================
# NVIDIA NGC MODEL INTEGRATION
# ============================================================================

class NVIDIALlamaClient:
    """Client for NVIDIA NGC hosted Llama model"""

    def __init__(self, api_key: str, endpoint: str):
        self.api_key = api_key
        self.endpoint = endpoint
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def generate_response(self, prompt: str, max_tokens: int = 1024,
                         temperature: float = 0.7) -> str:
        """Generate response from fine-tuned Llama model"""

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9,
            "stream": False
        }

        try:
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")

        except requests.exceptions.RequestException as e:
            print(f"⚠️ NGC API Error: {e}")
            return self._fallback_response()

    def _fallback_response(self) -> str:
        """Fallback response if API fails"""
        return json.dumps({
            "urgency_score": 8.5,
            "urgency_level": "CRITICAL",
            "call_number": "911",
            "recommended_action": "IMMEDIATE 911 - Suspected overdose",
            "immediate_instructions": [
                "Call 911 immediately",
                "Administer naloxone if available",
                "Place in recovery position",
                "Monitor breathing"
            ],
            "rationale": "API fallback - suspected overdose requires immediate response"
        })


# ============================================================================
# GEOSPATIAL UTILITIES
# ============================================================================

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates in miles (Haversine formula)"""
    R = 3959  # Earth's radius in miles

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (math.sin(delta_lat/2)**2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2)
    c = 2 * math.asin(math.sqrt(a))

    return R * c


def find_nearest_facilities(event_location: Location,
                            facilities_db: List[Dict],
                            facility_type: Optional[str] = None) -> List[Facility]:
    """Find nearest facilities to event location"""

    results = []

    for facility_data in facilities_db:
        # Parse location
        if 'location' not in facility_data:
            continue

        loc_data = facility_data['location']
        try:
            lat = float(loc_data['latitude'])
            lon = float(loc_data['longitude'])
        except (KeyError, ValueError):
            continue

        # Filter by type if specified
        if facility_type and facility_data.get('facility_type') != facility_type:
            continue

        # Calculate distance
        distance = calculate_distance(
            event_location.latitude, event_location.longitude,
            lat, lon
        )

        # Estimate response time (assume 25 mph average speed)
        estimated_time = (distance / 25.0) * 60  # minutes

        facility = Facility(
            name=facility_data.get('facility_name', 'Unknown'),
            facility_type=facility_data.get('facility_type', 'Unknown'),
            services=facility_data.get('services', 'Unknown'),
            location=Location(lat, lon,
                            facility_data.get('location', {}).get('human_address')),
            distance_miles=distance,
            estimated_response_time_min=estimated_time
        )

        results.append(facility)

    # Sort by distance
    results.sort(key=lambda f: f.distance_miles)

    return results


# ============================================================================
# SF HEALTH FACILITIES DATABASE (Sample - load from actual file)
# ============================================================================

# In production, load from the downloaded SF data
SF_FACILITIES = [
    {
        "facility_name": "California Pacific Med Ctr-pacific Campus",
        "facility_type": "General Acute Care Hospital",
        "services": "Hospital",
        "location": {
            "latitude": "37.79142444",
            "longitude": "-122.43103755",
            "human_address": "{\"address\": \"2333 BUCHANAN STREET\", \"city\": \"San Francisco\", \"state\": \"CA\"}"
        }
    },
    {
        "facility_name": "UCSF Medical Center",
        "facility_type": "General Acute Care Hospital",
        "services": "Hospital",
        "location": {
            "latitude": "37.76336734",
            "longitude": "-122.45856738",
            "human_address": "{\"address\": \"505 PARNASSUS AVENUE\", \"city\": \"San Francisco\", \"state\": \"CA\"}"
        }
    },
    {
        "facility_name": "Baart Turk Street Clinic",
        "facility_type": "Community Clinic",
        "services": "Drug Treatment",
        "location": {
            "latitude": "37.78246323",
            "longitude": "-122.41634657",
            "human_address": "{\"address\": \"433 TURK STREET\", \"city\": \"San Francisco\", \"state\": \"CA\"}"
        }
    },
    {
        "facility_name": "Walgreens Pharmacy #9342",
        "facility_type": "Pharmacy",
        "services": "24-hour Pharmacy",
        "location": {
            "latitude": "37.78503",
            "longitude": "-122.41234",
            "human_address": "{\"address\": \"825 MARKET STREET\", \"city\": \"San Francisco\", \"state\": \"CA\"}"
        }
    },
    {
        "facility_name": "CVS Pharmacy #9871",
        "facility_type": "Pharmacy",
        "services": "24-hour Pharmacy",
        "location": {
            "latitude": "37.78112",
            "longitude": "-122.42456",
            "human_address": "{\"address\": \"1524 POLK STREET\", \"city\": \"San Francisco\", \"state\": \"CA\"}"
        }
    }
]


# ============================================================================
# PROMPT ENGINEERING FOR FINE-TUNED MODEL
# ============================================================================

def build_emergency_prompt(event: EmergencyEvent,
                           nearest_facilities: Dict[str, List[Facility]]) -> str:
    """Build prompt for fine-tuned Llama model"""

    # Format symptoms
    symptoms_str = ", ".join(event.symptoms)

    # Format nearby facilities
    facilities_context = ""
    for ftype, facilities in nearest_facilities.items():
        if facilities:
            top_facility = facilities[0]
            facilities_context += f"\nNearest {ftype}: {top_facility.name} ({top_facility.distance_miles:.2f} miles, ~{top_facility.estimated_response_time_min:.1f} min)"

    prompt = f"""Instruction: Analyze this real-time emergency situation and provide an immediate response recommendation. Classify urgency (0-10 scale), determine if this requires 911 (urgent) or 311 (non-urgent), and provide specific instructions.

Input:
**Emergency Event:** {event.event_id}
**Time:** {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
**Location:** {event.location.latitude}, {event.location.longitude}
**Symptoms:** {symptoms_str}
**Patient Condition:** {event.patient_condition}
**Patient Age:** {event.patient_age or 'Unknown'}
**Witnessed Overdose:** {'Yes' if event.witnessed_overdose else 'No'}
**Naloxone Available:** {'Yes' if event.naloxone_available else 'No'}
**Caller Relation:** {event.caller_relation}
**Additional Context:** {event.additional_context}

**Nearby Resources:**{facilities_context}

**Historical Context (SF):**
- Average EMS response time: 3.67 minutes
- Weekly overdose calls: 86 average (2023)
- Naloxone distribution: 202,145 units/year (2024)

Output (JSON format):
Provide a structured response with:
1. urgency_score (0-10, where 10 = life-threatening)
2. urgency_level (CRITICAL/HIGH/MODERATE/LOW)
3. call_number (911 for urgent, 311 for non-urgent)
4. recommended_action (specific next steps)
5. immediate_instructions (list of actions caller should take NOW)
6. rationale (brief explanation of decision)
"""

    return prompt


# ============================================================================
# VAPI VOICE CALL INTEGRATION
# ============================================================================

class VAPIClient:
    """Client for VAPI voice calling service"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def make_call(self, phone_number: str, message: str,
                  urgency_level: str, event_id: str) -> Dict:
        """
        Initiate voice call via VAPI

        NOTE: This uses PLACEHOLDER numbers and will NOT actually call 311/911
        """

        # Determine voice urgency
        if "CRITICAL" in urgency_level or "911" in phone_number:
            voice_tone = "urgent"
            speaking_rate = 1.2  # Faster for emergencies
        else:
            voice_tone = "calm"
            speaking_rate = 1.0

        payload = {
            "phoneNumberId": "PLACEHOLDER-VAPI-NUMBER",
            "customer": {
                "number": phone_number  # PLACEHOLDER - not real 311/911
            },
            "assistant": {
                "model": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "messages": [
                        {
                            "role": "system",
                            "content": f"You are an emergency dispatch assistant. Speak in a {voice_tone} tone. This is a {urgency_level} situation."
                        }
                    ]
                },
                "voice": {
                    "provider": "11labs",
                    "voiceId": "21m00Tcm4TlvDq8ikWAM",  # Professional female voice
                    "stability": 0.8,
                    "similarityBoost": 0.75,
                    "speed": speaking_rate
                },
                "firstMessage": message,
                "endCallMessage": "Thank you for calling. Help is on the way.",
                "recordingEnabled": True,
                "metadata": {
                    "event_id": event_id,
                    "urgency_level": urgency_level,
                    "timestamp": datetime.now().isoformat()
                }
            }
        }

        try:
            print(f"\n📞 SIMULATED CALL to {phone_number}")
            print(f"   Message: {message[:100]}...")
            print(f"   [PLACEHOLDER - NOT ACTUALLY CALLING]\n")

            # In production, uncomment this to make real VAPI calls:
            # response = requests.post(
            #     VAPI_ENDPOINT,
            #     headers=self.headers,
            #     json=payload,
            #     timeout=10
            # )
            # response.raise_for_status()
            # return response.json()

            # Simulated response
            return {
                "success": True,
                "call_id": f"SIMULATED-{event_id}",
                "status": "initiated",
                "phone_number": phone_number,
                "message": "Call simulated successfully (placeholder mode)"
            }

        except Exception as e:
            print(f"⚠️ VAPI Error: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# ============================================================================
# MAIN EMERGENCY RESPONSE ORCHESTRATOR
# ============================================================================

class EmergencyResponseAgent:
    """Main agent orchestrating emergency response"""

    def __init__(self):
        self.llama_client = NVIDIALlamaClient(NGC_API_KEY, NGC_MODEL_ENDPOINT)
        self.vapi_client = VAPIClient(VAPI_API_KEY)
        self.facilities_db = SF_FACILITIES

    def process_emergency(self, event: EmergencyEvent) -> ResponseRecommendation:
        """Process emergency event and determine response"""

        print(f"\n{'='*70}")
        print(f"🚨 EMERGENCY EVENT: {event.event_id}")
        print(f"{'='*70}")
        print(f"Time: {event.timestamp}")
        print(f"Location: {event.location.latitude}, {event.location.longitude}")
        print(f"Symptoms: {', '.join(event.symptoms)}")
        print(f"Condition: {event.patient_condition}")
        print(f"Overdose witnessed: {event.witnessed_overdose}")
        print(f"Naloxone available: {event.naloxone_available}")

        # Step 1: Find nearest facilities
        print(f"\n📍 Finding nearest facilities...")
        nearest_pharmacy = find_nearest_facilities(
            event.location, self.facilities_db, "Pharmacy"
        )[:1]

        nearest_hospital = find_nearest_facilities(
            event.location, self.facilities_db, "General Acute Care Hospital"
        )[:1]

        nearest_clinic = find_nearest_facilities(
            event.location, self.facilities_db, "Community Clinic"
        )[:1]

        nearest_facilities = {
            "pharmacy": nearest_pharmacy,
            "hospital": nearest_hospital,
            "clinic": nearest_clinic
        }

        for ftype, facilities in nearest_facilities.items():
            if facilities:
                f = facilities[0]
                print(f"   {ftype.title()}: {f.name} ({f.distance_miles:.2f} mi)")

        # Step 2: Build prompt for fine-tuned model
        print(f"\n🤖 Consulting fine-tuned Llama model...")
        prompt = build_emergency_prompt(event, nearest_facilities)

        # Step 3: Get model recommendation
        model_response = self.llama_client.generate_response(prompt, max_tokens=1024)

        # Step 4: Parse model response
        try:
            # Extract JSON from response (model should return JSON)
            response_data = self._parse_model_response(model_response)
        except Exception as e:
            print(f"⚠️ Error parsing model response: {e}")
            response_data = self._default_critical_response()

        # Step 5: Build recommendation
        recommendation = ResponseRecommendation(
            urgency_score=response_data.get("urgency_score", 8.0),
            urgency_level=response_data.get("urgency_level", "CRITICAL"),
            call_number=PHONE_911_PLACEHOLDER if response_data.get("urgency_score", 8.0) >= URGENCY_SCORE_THRESHOLD
                       else PHONE_311_PLACEHOLDER,
            recommended_action=response_data.get("recommended_action", "Call 911 immediately"),
            nearest_pharmacy=nearest_pharmacy[0] if nearest_pharmacy else None,
            nearest_hospital=nearest_hospital[0] if nearest_hospital else None,
            nearest_clinic=nearest_clinic[0] if nearest_clinic else None,
            estimated_ems_arrival_min=3.67,  # SF average
            immediate_instructions=response_data.get("immediate_instructions", []),
            rationale=response_data.get("rationale", "")
        )

        # Step 6: Display recommendation
        self._display_recommendation(recommendation)

        # Step 7: Initiate call if appropriate
        if response_data.get("urgency_score", 0) >= 5.0:  # Moderate or higher
            self._initiate_emergency_call(event, recommendation)

        return recommendation

    def _parse_model_response(self, response: str) -> Dict:
        """Parse model response (expects JSON)"""
        # Try to extract JSON from response
        try:
            # Model might wrap JSON in markdown code blocks
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response

            return json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback: try to find JSON-like structure
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return self._default_critical_response()

    def _default_critical_response(self) -> Dict:
        """Default critical response if parsing fails"""
        return {
            "urgency_score": 9.0,
            "urgency_level": "CRITICAL",
            "call_number": "911",
            "recommended_action": "IMMEDIATE 911 - Call emergency services",
            "immediate_instructions": [
                "Call 911 immediately",
                "Stay with patient",
                "Monitor breathing",
                "Administer naloxone if available"
            ],
            "rationale": "Unable to parse model response - defaulting to critical"
        }

    def _display_recommendation(self, rec: ResponseRecommendation):
        """Display recommendation to console"""
        print(f"\n{'='*70}")
        print(f"📋 RECOMMENDATION")
        print(f"{'='*70}")
        print(f"Urgency Score: {rec.urgency_score}/10 ({rec.urgency_level})")
        print(f"Call Number: {rec.call_number}")
        print(f"Action: {rec.recommended_action}")

        if rec.immediate_instructions:
            print(f"\n🔴 IMMEDIATE INSTRUCTIONS:")
            for i, instruction in enumerate(rec.immediate_instructions, 1):
                print(f"   {i}. {instruction}")

        print(f"\n📍 NEAREST FACILITIES:")
        if rec.nearest_hospital:
            print(f"   Hospital: {rec.nearest_hospital.name} ({rec.nearest_hospital.distance_miles:.2f} mi)")
        if rec.nearest_clinic:
            print(f"   Clinic: {rec.nearest_clinic.name} ({rec.nearest_clinic.distance_miles:.2f} mi)")
        if rec.nearest_pharmacy:
            print(f"   Pharmacy: {rec.nearest_pharmacy.name} ({rec.nearest_pharmacy.distance_miles:.2f} mi)")

        if rec.estimated_ems_arrival_min:
            print(f"\n⏱️  Estimated EMS arrival: {rec.estimated_ems_arrival_min:.1f} minutes")

        print(f"\n💡 Rationale: {rec.rationale}")

    def _initiate_emergency_call(self, event: EmergencyEvent,
                                 rec: ResponseRecommendation):
        """Initiate VAPI call to 311 or 911 (PLACEHOLDER)"""

        # Build call message
        if rec.urgency_level == "CRITICAL":
            message = f"Emergency dispatch, this is an automated report for event {event.event_id}. "
            message += f"Critical situation at {event.location.address or 'location provided'}. "
            message += f"Patient is {event.patient_condition}, symptoms include {', '.join(event.symptoms[:3])}. "
            if event.witnessed_overdose:
                message += "Suspected drug overdose. "
            if event.naloxone_available:
                message += "Naloxone is available on scene. "
            message += f"Nearest hospital is {rec.nearest_hospital.name if rec.nearest_hospital else 'unknown'}, "
            message += f"{rec.nearest_hospital.distance_miles:.1f} miles away. "
            message += "Immediate medical response required."
        else:
            message = f"San Francisco 311, this is an automated health advisory for event {event.event_id}. "
            message += f"Non-emergency situation at {event.location.address or 'location provided'}. "
            message += f"Patient experiencing {', '.join(event.symptoms[:2])}. "
            message += f"Nearest clinic is {rec.nearest_clinic.name if rec.nearest_clinic else 'unknown'}. "
            message += "Requesting guidance or non-emergency resources."

        # Make call via VAPI
        call_result = self.vapi_client.make_call(
            phone_number=rec.call_number,
            message=message,
            urgency_level=rec.urgency_level,
            event_id=event.event_id
        )

        print(f"\n📞 Call Status: {call_result.get('status', 'Unknown')}")
        print(f"   Call ID: {call_result.get('call_id', 'N/A')}")


# ============================================================================
# DEMO / TESTING
# ============================================================================

def demo_overdose_emergency():
    """Demo: Suspected overdose emergency"""

    event = EmergencyEvent(
        event_id="OD-2026-001",
        timestamp=datetime.now(),
        location=Location(
            latitude=37.7838,
            longitude=-122.4167,
            address="455 Golden Gate Ave, San Francisco, CA"
        ),
        symptoms=[
            "unresponsive",
            "shallow breathing",
            "blue lips",
            "pinpoint pupils"
        ],
        patient_age=32,
        patient_condition="unconscious",
        witnessed_overdose=True,
        naloxone_available=True,
        caller_relation="bystander",
        additional_context="Found person collapsed in alley. Appears to be drug overdose. Needle nearby. Bystander has naloxone kit."
    )

    agent = EmergencyResponseAgent()
    recommendation = agent.process_emergency(event)

    return recommendation


def demo_non_urgent_pharmacy():
    """Demo: Non-urgent pharmacy need"""

    event = EmergencyEvent(
        event_id="PHARM-2026-001",
        timestamp=datetime.now(),
        location=Location(
            latitude=37.7749,
            longitude=-122.4194,
            address="Market St & 5th St, San Francisco, CA"
        ),
        symptoms=[
            "withdrawal symptoms",
            "anxiety",
            "nausea"
        ],
        patient_age=28,
        patient_condition="conscious",
        witnessed_overdose=False,
        naloxone_available=False,
        caller_relation="self",
        additional_context="Patient experiencing mild withdrawal. Needs to refill buprenorphine prescription. Ran out 2 days ago."
    )

    agent = EmergencyResponseAgent()
    recommendation = agent.process_emergency(event)

    return recommendation


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""

    print("\n" + "="*70)
    print("🚑 EMERGENCY RESPONSE AGENT")
    print("   NVIDIA NGC Fine-Tuned Llama + VAPI Integration")
    print("   SAFETY MODE: Using PLACEHOLDER phone numbers")
    print("="*70)

    # Demo 1: Critical overdose
    print("\n\n🔴 DEMO 1: CRITICAL OVERDOSE EMERGENCY\n")
    demo_overdose_emergency()

    print("\n\n" + "="*70 + "\n\n")

    # Demo 2: Non-urgent pharmacy
    print("\n\n🟡 DEMO 2: NON-URGENT PHARMACY NEED\n")
    demo_non_urgent_pharmacy()

    print("\n\n" + "="*70)
    print("✅ Demo complete. Review recommendations above.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
