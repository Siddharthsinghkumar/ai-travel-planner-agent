import sys
import os

# Ensure project root is on the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.planner_agent import plan_trip
from tools.booking_handoff import hold_booking, build_booking_handoff_url
import streamlit as st
from datetime import date

# --- Page Configuration & Custom Styles ---
st.set_page_config(page_title="AI Travel Planner", layout="centered", initial_sidebar_state="collapsed")

# Inject CSS for ChatGPT-like look and fully rounded chips & buttons
st.markdown(
    """
    <style>
    /* ─────────────────────────────────────────────────────────
       Container styling (centered, max-width = 700px)
    ───────────────────────────────────────────────────────── */
    .main {
        max-width: 700px;            /* match ChatGPT panel width */
        margin: 0 auto;              /* center on page */
        background-color: #0d1117;   /* dark background */
        color: #c9d1d9;              /* light text */
        padding: 2rem;               /* inner spacing */
        border-radius: 10px;         /* slight rounding */
    }

    /* ─────────────────────────────────────────────────────────
       Header styling
    ───────────────────────────────────────────────────────── */
    h1 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
        color: #58a6ff;              /* bright accent blue */
        text-align: center;
        margin-bottom: 1.5rem;
    }

    /* ─────────────────────────────────────────────────────────
       Pill-shaped input fields (text, date, textarea)
    ───────────────────────────────────────────────────────── */
    .stTextInput>div>div>input,
    .stDateInput>div>div>input,
    .stTextArea>div>div>textarea {
        background-color: #0d1117;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 9999px;       /* full pill shape */
        padding: 0.5rem 1rem;        /* comfy vertical + horizontal padding */
    }

    /* ─────────────────────────────────────────────────────────
       Pill-shaped select box
    ───────────────────────────────────────────────────────── */
    .stSelectbox>div>div>div>div {
        background-color: #0d1117;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 9999px;
        padding: 0.4rem 1rem;
    }
        /* make the row of chips one big capsule */
    .chips-container {
        display: inline-flex;
        flex-wrap: wrap;
        gap: 0.5rem;               /* space between chips */
        padding: 0.4rem;           /* inner padding of the capsule */
        background-color: #161b22; /* same as chip background */
        border: 1px solid #30363d; /* same border as chips */
        border-radius: 9999px;     /* full rounding of the container */
    }

    /* ─────────────────────────────────────────────────────────
       Send button styling (use st.button("Send") in code)
    ───────────────────────────────────────────────────────── */
    .chat-send-button {
        background-color: #238636;   /* green send button */
        color: #fff;
        border: none;
        border-radius: 9999px;
        padding: 0.5rem 1rem;
        margin-top: 0.5rem;
        cursor: pointer;
    }

    /* ─────────────────────────────────────────────────────────
       Recommendation "chips" (flight options)
    ───────────────────────────────────────────────────────── */
    .chip {
        display: inline-block;
        background-color: #21262d;   /* dark chip bg */
        color: #58a6ff;              /* accent blue text */
        border: 1px solid #30363d;
        border-radius: 9999px;       /* pill shape */
        padding: 0.4rem 1rem;
        margin: 0.3rem;
        cursor: pointer;
    }
    /* .chip:hover {
        background-color: #30363d;   /* slightly lighter on hover */
    } */

    /* ─────────────────────────────────────────────────────────
       Expanded recommendation details card
    ───────────────────────────────────────────────────────── */
    .expanded {
        padding: 20px;
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .expanded h3 {
        color: #58a6ff;              /* title accent */
    }

    /* ─────────────────────────────────────────────────────────
       Booking handoff section styling
    ───────────────────────────────────────────────────────── */
    .booking-handoff {
        background-color: #161b22;
        border: 2px solid #238636;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .booking-link {
        display: inline-block;
        background-color: #238636;
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 9999px;
        text-decoration: none;
        font-weight: 600;
        margin-top: 1rem;
        transition: background-color 0.3s;
    }
    .booking-link:hover {
        background-color: #2ea043;
        color: white;
        text-decoration: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Helper Functions ---
def recommendation_card(rec):
    """Render clickable recommendation chip"""
    label = f"✈️ {rec['airline']} · {rec['departure']} · Rs {rec['price']}"
    if st.button(label, key=f"rec_{rec['airline']}_{rec['departure']}"):
        st.session_state['selected_recommendation'] = rec

def show_expanded(rec):
    """Show expanded recommendation details"""
    st.markdown("---")
    st.markdown(
        f"""
        <div class='expanded'>
            <h3>{rec['airline']}</h3>
            <p><b>Departure:</b> {rec['departure']}</p>
            <p><b>Price:</b> Rs {rec['price']}</p>
            <p><b>Rating:</b> {'⭐' * int(rec['rating'])} ({rec['rating']})</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Close", key="close_rec"):
        st.session_state['selected_recommendation'] = None

def dedupe_flights(flights):
    """Remove duplicate flights based on airline, flight number, and date"""
    seen = set()
    unique = []
    for f in flights:
        # Create a unique key using airline, flight number, and departure date
        key = (f.get("airline"), f.get("flight_no"), f.get("date"))
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique

def display_booking_handoff(answer, origin, destination):
    """Display booking handoff section with hold and redirect functionality"""
    if answer.get("best_flight"):
        st.markdown('<div class="booking-handoff">', unsafe_allow_html=True)
        st.subheader("🎫 Ready to Book?")
        
        # Check if booking is already held in session state
        if 'booking_held' not in st.session_state:
            st.session_state.booking_held = False
            st.session_state.booking_id = None
            st.session_state.handoff_url = None
        
        if not st.session_state.booking_held:
            if st.button("✅ Hold This Flight (15 minutes)", type="primary"):
                try:
                    with st.spinner("Holding your flight..."):
                        # Hold the booking
                        booking_id = hold_booking(
                            flight=answer["best_flight"],
                            passenger={"name": "User"}
                        )
                        
                        # Build handoff URL
                        handoff_url = build_booking_handoff_url(
                            airline=answer["best_flight"]["airline"],
                            origin=origin,
                            destination=destination,
                            depart_date=answer["best_flight"]["date"]
                        )
                        
                        # Store in session state
                        st.session_state.booking_held = True
                        st.session_state.booking_id = booking_id
                        st.session_state.handoff_url = handoff_url
                        
                        st.success("✅ Flight held for 15 minutes")
                        st.rerun()  # Refresh to show the proceed button
                        
                except Exception as e:
                    st.error(f"❌ Failed to hold booking: {str(e)}")
        
        # If booking is held, show proceed button
        if st.session_state.booking_held:
            st.success(f"✅ Flight held (ID: {st.session_state.booking_id[:8]}...)")
            
            if st.session_state.handoff_url:
                st.markdown(
                    f'<a href="{st.session_state.handoff_url}" target="_blank" class="booking-link">🔗 Proceed to Booking</a>',
                    unsafe_allow_html=True
                )
                
                st.caption("⏰ Your seat is reserved for 15 minutes. Click above to complete booking on the airline's website.")
                
                # Option to release hold
                if st.button("🔄 Release Hold & Search Again"):
                    st.session_state.booking_held = False
                    st.session_state.booking_id = None
                    st.session_state.handoff_url = None
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# --- Main App ---
def main():
    st.title("✈️ AI Travel Planner")
    
    # Initialize session state for booking
    if 'booking_held' not in st.session_state:
        st.session_state.booking_held = False
        st.session_state.booking_id = None
        st.session_state.handoff_url = None
    
    # Use a form to prevent re-runs and flicker
    with st.form("trip_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            origin = st.text_input("🛫 From (IATA)", value="DEL")
        
        with col2:
            destination = st.text_input("🌍 Destination (IATA)", value="BOM")
        
        col3, col4 = st.columns(2)
        
        with col3:
            travel_date = st.date_input("📅 Travel Date", value=date.today())
        
        with col4:
            trip_type = st.selectbox("✈️ Trip Type", 
                                   ["Business", "Urgent", "Holiday", "Flexible"],
                                   index=0)
        
        user_query = st.text_area("💬 Ask anything about your trip...", 
                                height=100,
                                placeholder="Example: I need a morning flight with no layovers under ₹5000")
        
        submitted = st.form_submit_button("🚀 Plan My Trip")
    
    # Handle form submission
    if submitted:
        # Reset booking state on new search
        st.session_state.booking_held = False
        st.session_state.booking_id = None
        st.session_state.handoff_url = None
        
        try:
            with st.spinner("🔍 Finding the best flights for you..."):
                answer = plan_trip(
                    origin=origin,
                    destination=destination,
                    date=travel_date.strftime("%Y-%m-%d"),
                    user_query=user_query,
                    trip_type=trip_type
                )
            
            # Store answer in session state for potential future use
            st.session_state['last_answer'] = answer
            st.session_state['last_origin'] = origin
            st.session_state['last_destination'] = destination
            
            if "error" in answer:
                st.error(f"❌ {answer['error']}")
            else:
                st.success("✅ Best flight found!")
                
                # Display AI response
                st.markdown(f"**🤖 AI Travel Agent:** {answer['llm_response']}")
                
                # Display weather if available - with friendlier error handling
                if answer.get('weather'):
                    weather_text = answer['weather']
                    # Check if weather starts with error emoji (❌)
                    if isinstance(weather_text, str) and weather_text.startswith("❌"):
                        st.info("🌤️ Weather forecast unavailable for this date. Showing flight recommendations only.")
                    else:
                        st.info(f"🌤️ Weather at destination: {weather_text}")
                
                # Display best flight details
                if answer.get("best_flight"):
                    st.subheader("✨ Best Flight Recommendation")
                    st.json(answer["best_flight"])
                
                # Display filtered flights if available (with deduplication)
                if answer.get("filtered_flights"):
                    # Deduplicate flights before displaying
                    unique_flights = dedupe_flights(answer["filtered_flights"])
                    
                    st.subheader("📋 Other Matching Flights")
                    for flight in unique_flights[:5]:  # Show top 5 unique flights
                        with st.expander(f"✈️ {flight.get('airline', 'Unknown')} - ₹{flight.get('price_inr', 'N/A')}"):
                            st.write(f"**Flight:** {flight.get('flight_no', 'N/A')}")
                            st.write(f"**Departure:** {flight.get('departure_time', 'N/A')}")
                            st.write(f"**Arrival:** {flight.get('arrival_time', 'N/A')}")
                            st.write(f"**Duration:** {flight.get('duration', 'N/A')}")
                            st.write(f"**Baggage:** {flight.get('baggage', 'N/A')}")
                    
                    # Show count of unique flights found
                    if len(unique_flights) > 5:
                        st.caption(f"*Showing 5 of {len(unique_flights)} unique flights*")
                    else:
                        st.caption(f"*Found {len(unique_flights)} unique flights*")
                
                # Show fallback note if present
                if answer.get("fallback_note"):
                    st.warning(f"⚠️ {answer['fallback_note']}")
                
                # Display booking handoff section
                display_booking_handoff(answer, origin, destination)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)
    
    # Display previous results if available (not in form submission mode)
    elif 'last_answer' in st.session_state:
        answer = st.session_state['last_answer']
        
        if not answer.get("error"):
            st.info("ℹ️ Showing previous search results")
            
            if answer.get("best_flight"):
                st.subheader("✨ Your Selected Flight")
                st.json(answer["best_flight"])
            
            if answer.get("filtered_flights"):
                # Deduplicate previous flights too
                unique_flights = dedupe_flights(answer["filtered_flights"])
                
                st.subheader("📋 Previous Flight Options")
                for flight in unique_flights[:3]:  # Show top 3 unique flights
                    st.write(f"**{flight.get('airline', 'Unknown')}** - ₹{flight.get('price_inr', 'N/A')}")
                    st.write(f"Departure: {flight.get('departure_time', 'N/A')}")
                    st.write("---")
            
            # Show booking handoff for previous search if available
            if st.session_state.get('last_origin') and st.session_state.get('last_destination'):
                display_booking_handoff(
                    answer, 
                    st.session_state['last_origin'], 
                    st.session_state['last_destination']
                )

if __name__ == "__main__":
    main()