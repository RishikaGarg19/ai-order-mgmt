

# import streamlit as st
# import pandas as pd
# import os
# from dotenv import load_dotenv
# from data_analyzer import DataAnalyzer
# from chat_handler import ChatHandler
# from visualization import Visualizer
# from alert_system import AlertSystem
# # Import the production-optimized ML predictor
# from ml_predictor import OrderPrioritizationAI
# import datetime
# import traceback

# # Load environment variables
# load_dotenv()

# # Page configuration
# st.set_page_config(
#     page_title="Enterprise Order Analytics Platform",
#     page_icon="📊",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Professional CSS styling for enterprise applications
# st.markdown("""
# <style>
#     .main > div {
#         padding-top: 2rem;
#     }
    
#     .stApp {
#         background-color: #F8F9FA;
#     }
    
#     .css-1d391kg {
#         background-color: #FFFFFF;
#     }
    
#     /* Remove unwanted white space/bars */
#     .block-container {
#         padding-top: 1rem;
#         padding-bottom: 0rem;
#     }
    
#     /* Hide Streamlit style elements */
#     #MainMenu {visibility: hidden;}
#     .stDeployButton {display:none;}
#     footer {visibility: hidden;}
#     .stApp > header {visibility: hidden;}
    
#     /* Fix for random white bars */
#     div[data-testid="stToolbar"] {
#         visibility: hidden;
#         height: 0%;
#         position: fixed;
#     }
    
#     div[data-testid="stDecoration"] {
#         visibility: hidden;
#         height: 0%;
#         position: fixed;
#     }
    
#     div[data-testid="stStatusWidget"] {
#         visibility: hidden;
#         height: 0%;
#         position: fixed;
#     }
    
#     .stSidebar .stButton > button {
#         background-color: #2C3E50;
#         color: white;
#         border: none;
#         border-radius: 6px;
#         font-weight: 500;
#         margin-bottom: 8px;
#         padding: 0.6rem 1rem;
#         font-size: 0.85rem;
#         transition: all 0.3s ease;
#         width: 100%;
#         text-align: left;
#     }
    
#     .stSidebar .stButton > button:hover {
#         background-color: #1A252F;
#         color: white;
#         transform: translateY(-1px);
#         box-shadow: 0 3px 6px rgba(0,0,0,0.15);
#     }
    
#     .chat-container {
#         background-color: white;
#         border-radius: 8px;
#         padding: 1.5rem;
#         box-shadow: 0 2px 8px rgba(0,0,0,0.1);
#         margin-bottom: 1rem;
#         max-height: 80vh;
#         overflow-y: auto;
#     }
    
#     .main-header {
#         color: #2C3E50;
#         font-size: 2.5rem;
#         font-weight: 600;
#         margin-bottom: 0.5rem;
#         font-family: 'Segoe UI', sans-serif;
#     }
    
#     .sub-header {
#         color: #5D6D7E;
#         font-size: 1.1rem;
#         margin-bottom: 2rem;
#         font-family: 'Segoe UI', sans-serif;
#     }
    
#     .section-header {
#         color: #2C3E50;
#         font-size: 1.2rem;
#         font-weight: 600;
#         margin-bottom: 1rem;
#         font-family: 'Segoe UI', sans-serif;
#     }
    
#     .standard-query {
#         background-color: #E3F2FD;
#         border-left: 4px solid #1976D2;
#         padding: 0.5rem;
#         margin: 0.5rem 0;
#         border-radius: 4px;
#     }
    
#     .custom-query {
#         background-color: #F3E5F5;
#         border-left: 4px solid #7B1FA2;
#         padding: 0.5rem;
#         margin: 0.5rem 0;
#         border-radius: 4px;
#     }
    
#     .ai-query {
#         background-color: #E8F5E8;
#         border-left: 4px solid #27AE60;
#         padding: 0.5rem;
#         margin: 0.5rem 0;
#         border-radius: 4px;
#     }
    
#     .info-box {
#         background-color: #E8F4FD;
#         border: 1px solid #2E86AB;
#         border-radius: 8px;
#         padding: 1rem;
#         margin: 1rem 0;
#     }
    
#     .success-box {
#         background-color: #E8F5E8;
#         border: 1px solid #27AE60;
#         border-radius: 8px;
#         padding: 1rem;
#         margin: 1rem 0;
#     }
    
#     .warning-box {
#         background-color: #FFF3CD;
#         border: 1px solid #FFC107;
#         border-radius: 8px;
#         padding: 1rem;
#         margin: 1rem 0;
#     }
    
#     .alert-critical {
#         background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
#         color: white;
#         padding: 0.8rem;
#         border-radius: 6px;
#         margin: 0.3rem 0;
#         box-shadow: 0 2px 6px rgba(231, 76, 60, 0.3);
#         font-size: 0.85rem;
#         font-weight: 500;
#     }
    
#     .alert-warning {
#         background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
#         color: white;
#         padding: 0.8rem;
#         border-radius: 6px;
#         margin: 0.3rem 0;
#         box-shadow: 0 2px 6px rgba(243, 156, 18, 0.3);
#         font-size: 0.85rem;
#         font-weight: 500;
#     }
    
#     .alert-info {
#         background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
#         color: white;
#         padding: 0.8rem;
#         border-radius: 6px;
#         margin: 0.3rem 0;
#         box-shadow: 0 2px 6px rgba(52, 152, 219, 0.3);
#         font-size: 0.85rem;
#         font-weight: 500;
#     }
    
#     .model-status-optimized {
#         background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
#         color: white;
#         padding: 1rem;
#         border-radius: 8px;
#         margin: 0.8rem 0;
#         box-shadow: 0 4px 12px rgba(30, 60, 114, 0.3);
#         border-left: 5px solid #ffffff;
#     }
    
#     .model-status-ready {
#         background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
#         color: white;
#         padding: 1rem;
#         border-radius: 8px;
#         margin: 0.8rem 0;
#         box-shadow: 0 4px 12px rgba(39, 174, 96, 0.3);
#         border-left: 5px solid #ffffff;
#     }
    
#     .model-status-training {
#         background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
#         color: white;
#         padding: 1rem;
#         border-radius: 8px;
#         margin: 0.8rem 0;
#         box-shadow: 0 4px 12px rgba(243, 156, 18, 0.3);
#         border-left: 5px solid #ffffff;
#     }
    
#     .performance-indicator {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         padding: 0.4rem 0.8rem;
#         border-radius: 15px;
#         font-size: 0.75rem;
#         font-weight: 600;
#         margin: 0.2rem 0;
#         display: inline-block;
#         box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3);
#     }
    
#     .logo-container {
#         display: flex;
#         justify-content: space-between;
#         align-items: center;
#         margin-bottom: 2rem;
#         padding: 1rem 0.5rem;
#     }
    
#     .logo-image {
#         max-width: 120px;
#         max-height: 80px;
#         object-fit: contain;
#     }
    
#     .placeholder-questions {
#         background-color: #F8F9FA;
#         border-radius: 8px;
#         padding: 1rem;
#         margin: 1rem 0;
#         border: 1px solid #E8E9EA;
#     }
    
#     .placeholder-question-button {
#         background-color: #2C3E50;
#         color: white;
#         border: none;
#         border-radius: 6px;
#         padding: 0.8rem 1rem;
#         margin: 0.5rem 0;
#         width: 100%;
#         font-size: 0.9rem;
#         font-weight: 500;
#         cursor: pointer;
#         transition: all 0.3s ease;
#     }
    
#     .placeholder-question-button:hover {
#         background-color: #1A252F;
#         transform: translateY(-1px);
#         box-shadow: 0 3px 6px rgba(0,0,0,0.15);
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if 'messages' not in st.session_state:
#     st.session_state.messages = []
# if 'data_loaded' not in st.session_state:
#     st.session_state.data_loaded = False
# if 'show_dashboard' not in st.session_state:
#     st.session_state.show_dashboard = False
# if 'current_analysis' not in st.session_state:
#     st.session_state.current_analysis = None
# if 'show_analytics_modal' not in st.session_state:
#     st.session_state.show_analytics_modal = False
# if 'ml_predictor' not in st.session_state:
#     st.session_state.ml_predictor = None
# if 'alert_system' not in st.session_state:
#     st.session_state.alert_system = None

# def format_large_number(value):
#     """Format large numbers with K, M, B suffixes"""
#     if pd.isna(value) or value == 0:
#         return "0"
    
#     if abs(value) >= 1_000_000_000:
#         return f"{value/1_000_000_000:.1f}B"
#     elif abs(value) >= 1_000_000:
#         return f"{value/1_000_000:.1f}M"
#     elif abs(value) >= 1_000:
#         return f"{value/1_000:.1f}K"
#     else:
#         return f"{value:,.0f}"

# def format_currency(value):
#     """Format currency values with K, M, B suffixes"""
#     if pd.isna(value) or value == 0:
#         return "$0"
    
#     if abs(value) >= 1_000_000_000:
#         return f"${value/1_000_000_000:.1f}B"
#     elif abs(value) >= 1_000_000:
#         return f"${value/1_000_000:.1f}M"
#     elif abs(value) >= 1_000:
#         return f"${value/1_000:.1f}K"
#     else:
#         return f"${value:,.0f}"

# @st.cache_data
# def load_data():
#     """Load and cache the enterprise dataset"""
#     try:
#         possible_paths = [
#             'data/orders.csv',
#             'orders.csv',
#             'data/dataset.csv'
#         ]
        
#         df = None
#         for path in possible_paths:
#             if os.path.exists(path):
#                 df = pd.read_csv(path)
#                 break
        
#         if df is None:
#             st.error("Dataset not found. Please upload your CSV file.")
#             return None
            
#         # Convert date columns
#         date_columns = ['order_date', 'requested_delivery_date', 'planned_delivery_date', 
#                        'actual_delivery_date', 'cancellation_date']
        
#         for col in date_columns:
#             if col in df.columns:
#                 df[col] = pd.to_datetime(df[col], errors='coerce')
        
#         return df
#     except Exception as e:
#         st.error(f"Error loading data: {str(e)}")
#         return None

# def initialize_ml_predictor(df):
#     """Initialize production-optimized ML predictor with pre-calculated cache"""
#     try:
#         if st.session_state.ml_predictor is None:
#             with st.spinner("Initializing AI-driven analytics system (initial setup: 2-3 minutes)..."):
#                 st.session_state.ml_predictor = OrderPrioritizationAI(df)
#                 st.success("AI analytics system operational. Subsequent analysis will execute instantly.")
#         return st.session_state.ml_predictor
#     except Exception as e:
#         st.error(f"Error initializing ML predictor: {str(e)}")
#         return None

# def initialize_alert_system(df):
#     """Initialize enterprise alert monitoring system"""
#     try:
#         if st.session_state.alert_system is None:
#             st.session_state.alert_system = AlertSystem(df)
#         return st.session_state.alert_system
#     except Exception as e:
#         st.error(f"Error initializing alert system: {str(e)}")
#         return None

# def display_alerts(alert_system):
#     """Display enterprise alerts in sidebar with critical alerts first"""
#     st.markdown('<h3 class="section-header">Enterprise Alert Center</h3>', unsafe_allow_html=True)
    
#     alerts = alert_system.generate_alerts()
#     alert_summary = alert_system.get_alert_summary()
    
#     # Alert summary metrics
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Critical", alert_summary['critical'], delta=None)
#     with col2:
#         st.metric("Warning", alert_summary['warning'], delta=None)
#     with col3:
#         st.metric("Info", alert_summary['info'], delta=None)
    
#     # Display alerts
#     if len(alerts) == 0:
#         st.success("All systems operating within normal parameters")
#     else:
#         # Filter and display critical alerts first
#         critical_alerts = [alert for alert in alerts if alert['severity'] == 'critical']
#         non_critical_alerts = [alert for alert in alerts if alert['severity'] != 'critical']
        
#         # Show critical alerts first
#         for alert in critical_alerts[:5]:
#             severity_class = f"alert-{alert['severity']}"
#             alert_text = alert['message']
#             st.markdown(f'<div class="{severity_class}">{alert_text}</div>', unsafe_allow_html=True)
        
#         # If there are more alerts (warning/info), show expand option
#         if len(non_critical_alerts) > 0:
#             remaining_count = len(non_critical_alerts)
#             st.info(f"Showing critical alerts. {remaining_count} warning/info alerts available")
            
#             if st.button("View All Alerts", key="view_all_alerts"):
#                 with st.expander("Warning and Info Alerts", expanded=True):
#                     for alert in non_critical_alerts:
#                         severity_class = f"alert-{alert['severity']}"
#                         st.markdown(f'<div class="{severity_class}">{alert["message"]}</div>', unsafe_allow_html=True)

# def is_ai_prioritization_query(user_query):
#     """Check if query specifically requests AI prioritization"""
#     ai_prioritization_keywords = [
#         'ai prioritization', 'ai-powered prioritization', 'ml prioritization',
#         'advanced ai prioritization', 'predictive prioritization', 
#         'ai-driven prioritization', 'machine learning prioritization',
#         'strategic prioritization with intervention', 'ai priority analysis',
#         'ai powered prioritization', 'generate ai powered prioritization',
#         'ai prioritization analysis', 'ml priority', 'ai prioritized orders',
#         'show me ai prioritized orders', 'ai prioritized', 'prioritized orders',
#         'ai models', 'machine learning models', 'ml models'
#     ]
    
#     query_lower = user_query.lower()
#     return any(keyword in query_lower for keyword in ai_prioritization_keywords)

# def display_placeholder_questions(latest_month_year):
#     """Display placeholder questions in main interface"""
#     st.markdown("### Get started with these common queries:")
    
#     # Placeholder questions
#     placeholder_questions = [
#         f"Executive summary for {latest_month_year}",
#         # f"Revenue performance analysis for {latest_month_year}",
#         "Top operational challenges analysis",
#         f"Advanced AI prioritization for {latest_month_year}"
#     ]
    
#     # Create buttons for each question
#     for i, question in enumerate(placeholder_questions):
#         if st.button(question, key=f"placeholder_q_{i}", use_container_width=True):
#             st.session_state.messages.append({
#                 "role": "user", 
#                 "content": question,
#                 "is_standard": True
#             })
#             st.rerun()

# def process_user_message(user_query, df, analyzer, chat_handler, is_standard=False):
#     """Process user message and generate intelligent response"""
#     try:
#         print(f"Processing message: '{user_query}' (standard: {is_standard})")
        
#         # For standard queries (button clicks), use the structured analysis
#         if is_standard:
#             # Analyze query requirements
#             analysis_result = analyzer.analyze_query(user_query)
            
#             # Check if this requires ML processing
#             if analysis_result.get("type") == "ai_prioritization_request":
#                 # Use the production-optimized ML predictor
#                 ml_predictor = initialize_ml_predictor(df)
#                 if ml_predictor:
#                     with st.spinner("Executing advanced analytics pipeline..."):
#                         st.info("Retrieving pre-calculated intelligence for instantaneous response")
#                         ai_results = ml_predictor.run_full_prioritization(current_month_only=True, top_n=20)
                        
#                         # Store results for analytics
#                         st.session_state.current_analysis = {
#                             'query': user_query,
#                             'result': ai_results,
#                             'is_standard': is_standard,
#                             'type': 'ai_prioritization'
#                         }
                        
#                         # Format AI response
#                         if 'error' in ai_results:
#                             return f"Analytics Error: {ai_results['error']}"
#                         else:
#                             return format_ai_prioritization_response(ai_results)
#                 else:
#                     return "Unable to initialize AI analytics system. Please verify data integrity and try again."
            
#             # For other standard queries, use standard analyzer
#             print(f"Analysis result type: {analysis_result.get('type', 'unknown')}")
            
#             # Store current analysis for analytics
#             st.session_state.current_analysis = {
#                 'query': user_query,
#                 'result': analysis_result,
#                 'is_standard': is_standard
#             }
            
#             # Generate intelligent response
#             response = chat_handler.get_response(user_query, analysis_result, df, is_placeholder=is_standard)
#             print(f"Response generated successfully")
            
#             return response
        
#         # For typed queries, check if it's AI prioritization
#         elif is_ai_prioritization_query(user_query):
#             # Use the production-optimized ML predictor for AI prioritization
#             ml_predictor = initialize_ml_predictor(df)
#             if ml_predictor:
#                 with st.spinner("Executing advanced analytics pipeline..."):
#                     st.info("Retrieving pre-calculated intelligence for instantaneous response")
#                     ai_results = ml_predictor.run_full_prioritization(current_month_only=True, top_n=20)
                    
#                     # Store results for analytics
#                     st.session_state.current_analysis = {
#                         'query': user_query,
#                         'result': ai_results,
#                         'is_standard': False,
#                         'type': 'ai_prioritization'
#                     }
                    
#                     # Format AI response
#                     if 'error' in ai_results:
#                         return f"Analytics Error: {ai_results['error']}"
#                     else:
#                         return format_ai_prioritization_response(ai_results)
#             else:
#                 return "Unable to initialize AI analytics system. Please verify data integrity and try again."
        
#         # For all other typed queries, use dynamic pandas analysis through OpenAI
#         else:
#             print(f"Processing typed query through OpenAI: {user_query}")
            
#             # Analyze query to get basic structure
#             analysis_result = analyzer.analyze_query(user_query)
            
#             # Store current analysis for analytics
#             st.session_state.current_analysis = {
#                 'query': user_query,
#                 'result': analysis_result,
#                 'is_standard': False,
#                 'type': 'dynamic'
#             }
            
#             # Generate response through OpenAI with pandas analysis
#             response = chat_handler.get_response(user_query, analysis_result, df, is_placeholder=False)
#             print(f"Dynamic response generated successfully")
            
#             return response
        
#     except Exception as e:
#         error_msg = f"Analysis encountered an error: {str(e)}"
#         print(f"Error processing message: {str(e)}")
#         st.error(error_msg)
#         st.error("Detailed error:")
#         st.code(traceback.format_exc())
#         return error_msg

# def format_ai_prioritization_response(ai_results):
#     """Format AI prioritization results into executive summary"""
#     try:
#         response = f"""
# # ADVANCED ANALYTICS: ORDER PRIORITIZATION GUIDANCE



# ## Strategically Prioritized Orders 
# """
        
#         # Add top 5 orders details
#         top_priorities = ai_results.get('top_priorities', pd.DataFrame())
#         if not top_priorities.empty:
#             for idx, row in top_priorities.head(10).iterrows():
#                 response += f"""
# ### Priority Rank #{idx+1}: Order {row.get('order_id', 'N/A')}
# - **Account**: {row.get('customer_name', 'N/A')} ({row.get('customer_tier', 'N/A')})
# - **Order Value**: ${row.get('order_value_usd', 0):,.0f}
# - **Priority Index**: {row.get('priority_score', 0):.1f}/100
# - **Risk Assessment**: {row.get('predicted_delay_probability', 0):.1%} probability
# - **Strategic Rationale**: {row.get('priority_reason', 'N/A')}
# """
        
#         # Add AI interventions - RESTORED FUNCTIONALITY
#         if ai_results.get('ai_interventions'):
#             response += "\n## AI-Recommended Interventions\n"
#             for i, intervention in enumerate(ai_results['ai_interventions'][:5]):
#                 response += f"\n### Order {intervention['order_id']} - {intervention['customer']}\n"
#                 response += f"- **Priority Score**: {intervention['priority_score']:.1f}/100\n"
#                 response += f"- **AI Delay Risk**: {intervention['ai_delay_risk']:.1%}\n"
#                 response += f"- **Predicted Delay**: {intervention['ai_predicted_days']:.0f} days\n"
#                 response += "\n**Recommended Actions**:\n"
#                 for action in intervention['interventions']:
#                     response += f"- {action}\n"
#                 response += "\n---\n"
        
#             response += f"""

# {ai_results.get('executive_summary', 'Analysis completed successfully')}

#  """
        
#         return response
        
#     except Exception as e:
#         return f"Advanced analytics completed successfully, but formatting encountered an issue: {str(e)}"

# def has_pending_user_message():
#     """Check if there's a user message that needs a response"""
#     if len(st.session_state.messages) == 0:
#         return False
    
#     last_message = st.session_state.messages[-1]
#     if last_message["role"] != "user":
#         return False
    
#     return len(st.session_state.messages) % 2 == 1

# def get_latest_data_date(df):
#     """Get the latest date available in the dataset"""
#     if 'order_date' in df.columns:
#         return df['order_date'].max()
#     return datetime.datetime(2025, 5, 31)  # Default to May 2025

# def show_model_status():
#     """Show AI model status and system controls"""
#     st.markdown('<h3 class="section-header">AI Analytics System Status</h3>', unsafe_allow_html=True)
    
#     if st.session_state.ml_predictor:
#         model_info = st.session_state.ml_predictor.get_model_status()
        
#         if model_info.get('models_loaded', False) and model_info.get('cache_loaded', False):
#             model_age = model_info.get('model_age_hours', 0)
#             cached_orders = model_info.get('cached_orders', 0)
            
#             st.markdown(f"""
#             <div class="model-status-optimized">
#                 <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">AI Analytics System: Operational</div>
#                 <div style="font-size: 1.4rem; font-weight: 700; margin: 0.5rem 0;">INSTANT PROCESSING ENABLED</div>
#                 <div style="font-size: 0.9rem; opacity: 0.9;">
#                     System Age: {model_age:.1f}h • Cache: {cached_orders:,} orders • Performance: {model_info.get('performance', 'Optimized')}
#                 </div>
#                 <div style="font-size: 0.8rem; margin-top: 0.5rem; opacity: 0.8;">
#                     Location: {model_info.get('models_folder', 'saved_models')} • Production Status: Operational
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)
#         elif model_info.get('models_loaded', False):
#             st.markdown(f"""
#             <div class="model-status-ready">
#                 <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">AI Models: Ready</div>
#                 <div style="font-size: 1.4rem; font-weight: 700; margin: 0.5rem 0;">CACHE INITIALIZATION...</div>
#                 <div style="font-size: 0.9rem; opacity: 0.9;">Next query will activate optimized processing</div>
#             </div>
#             """, unsafe_allow_html=True)
#         else:
#             st.markdown(f"""
#             <div class="model-status-training">
#                 <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">AI Models</div>
#                 <div style="font-size: 1.4rem; font-weight: 700; margin: 0.5rem 0;">READY FOR INITIALIZATION</div>
#                 <div style="font-size: 0.9rem; opacity: 0.9;">First query: 2-3 min setup, then optimized processing</div>
#             </div>
#             """, unsafe_allow_html=True)
#     else:
#         st.markdown(f"""
#         <div class="model-status-training">
#             <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">AI Analytics System</div>
#             <div style="font-size: 1.4rem; font-weight: 700; margin: 0.5rem 0;">NOT INITIALIZED</div>
#             <div style="font-size: 0.9rem; opacity: 0.9;">Will activate on first AI query</div>
#         </div>
#         """, unsafe_allow_html=True)
    
#     # System controls
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("Retrain Models", help="Retrain all AI models and rebuild cache"):
#             if st.session_state.ml_predictor:
#                 with st.spinner("Retraining models and rebuilding cache..."):
#                     st.session_state.ml_predictor.force_retrain_models()
#                 st.success("Models retrained and cache rebuilt successfully")
#                 st.rerun()
#             else:
#                 st.warning("Initialize ML predictor first by running an AI query")
    
#     with col2:
#         if st.button("System Details", help="Show detailed model and cache information"):
#             if st.session_state.ml_predictor:
#                 model_info = st.session_state.ml_predictor.get_model_status()
#                 st.json(model_info)
#             else:
#                 st.info("No ML predictor initialized yet")

# def show_analytics_modal():
#     """Display analytics in executive dashboard format"""
#     if st.session_state.show_analytics_modal and st.session_state.current_analysis:
        
#         with st.container():
#             st.markdown("---")
#             st.markdown("### Executive Analytics Dashboard")
            
#             # Close button
#             col1, col2 = st.columns([6, 1])
#             with col2:
#                 if st.button("Close", key="close_analytics"):
#                     st.session_state.show_analytics_modal = False
#                     st.rerun()
            
#             with col1:
#                 query = st.session_state.current_analysis['query']
#                 analysis_result = st.session_state.current_analysis['result']
#                 is_standard = st.session_state.current_analysis['is_standard']
#                 analysis_type = st.session_state.current_analysis.get('type', 'standard')
                
#                 # Show query type
#                 if analysis_type == 'ai_prioritization':
#                     st.success("Advanced AI Prioritization Analytics")
#                 elif is_standard:
#                     st.success("Standard Business Analytics")
#                 else:
#                     st.info("Custom Intelligence Analysis")
                
#                 st.markdown(f"**Executive Query:** {query}")
                
#                 # Initialize visualizer with current data
#                 if 'df' in st.session_state and st.session_state.df is not None:
#                     visualizer = Visualizer(st.session_state.df)
                    
#                     # Create enhanced visualizations
#                     try:
#                         visualizer.create_enhanced_visualizations(query, analysis_result)
                        
#                         # # Show AI prioritization summary if applicable
#                         # if analysis_type == 'ai_prioritization' or 'total_orders' in analysis_result:
#                         #     st.markdown("### AI Intelligence Summary")
                            
#                         #     col1, col2, col3, col4 = st.columns(4)
#                         #     with col1:
#                         #         st.metric(
#                         #             "High Priority Orders", 
#                         #             format_large_number(analysis_result.get('high_priority_count', 0))
#                         #         )
#                         #     with col2:
#                         #         st.metric(
#                         #             "Critical Risk Orders", 
#                         #             format_large_number(analysis_result.get('critical_risk_count', 0))
#                         #         )
#                         #     with col3:
#                         #         total_orders = analysis_result.get('total_orders', 1)
#                         #         risk_pct = (analysis_result.get('critical_risk_count', 0) / total_orders * 100) if total_orders > 0 else 0
#                         #         st.metric("Risk Percentage", f"{risk_pct:.1f}%")
#                         #     with col4:
#                         #         st.metric("Processing Time", analysis_result.get('processing_time', 'Optimized'))
                        
#                     except Exception as viz_error:
#                         st.error(f"Visualization error: {str(viz_error)}")
#                         st.info("Analysis completed successfully, but charts could not be generated.")

# def main():
#     # Store df in session state for analytics
#     df = None
    
#     # Header
#     st.markdown('<h1 class="main-header">Enterprise Order Management Platform</h1>', unsafe_allow_html=True)
#     st.markdown('<p class="sub-header">Advanced AI-driven business intelligence with optimization support​</p>', unsafe_allow_html=True)
    
#     # Sidebar
#     with st.sidebar:
#         # Company logos at the very top with balanced sizing
#         col1, col2 = st.columns([1.2, 1])
#         with col1:
#             try:
#                 if os.path.exists("images/AI Foundry Logo.png"):
#                     st.markdown(
#                         '<div style="display: flex; align-items: center; justify-content: center; height: 70px;">',
#                         unsafe_allow_html=True
#                     )
#                     st.image("images/AI Foundry Logo.png", width=250)
#                     st.markdown('</div>', unsafe_allow_html=True)
#                 else:
#                     st.markdown('<div style="display: flex; align-items: center; justify-content: center; height: 80px; text-align: center; font-size: 1.3rem; font-weight: bold; color: #2C3E50;">AI FOUNDRY</div>', unsafe_allow_html=True)
#             except Exception as e:
#                 st.markdown('<div style="display: flex; align-items: center; justify-content: center; height: 80px; text-align: center; font-size: 1.3rem; font-weight: bold; color: #2C3E50;">AI FOUNDRY</div>', unsafe_allow_html=True)

#         with col2:
#             try:
#                 if os.path.exists("images/international.png"):
#                     st.markdown(
#                         '<div style="display: flex; align-items: center; justify-content: center; height: 60px;">',
#                         unsafe_allow_html=True
#                     )
#                     st.image("images/international.png", width=230)
#                     st.markdown('</div>', unsafe_allow_html=True)
#                 else:
#                     st.markdown('<div style="display: flex; align-items: center; justify-content: center; height: 80px; text-align: center; font-size: 1.2rem; font-weight: bold; color: #2C3E50;">INTERNATIONAL</div>', unsafe_allow_html=True)
#             except Exception as e:
#                 st.markdown('<div style="display: flex; align-items: center; justify-content: center; height: 80px; text-align: center; font-size: 1.2rem; font-weight: bold; color: #2C3E50;">INTERNATIONAL</div>', unsafe_allow_html=True)        
#         # Minimal spacing after logos
#         st.markdown('<div style="margin-bottom: 0.8rem;"></div>', unsafe_allow_html=True)
        
#         st.markdown('<h2 class="section-header">Data Configuration</h2>', unsafe_allow_html=True)
        
#         # File upload option
#         uploaded_file = st.file_uploader("Upload Enterprise Data", type=['csv'])
        
#         if uploaded_file is not None:
#             df = pd.read_csv(uploaded_file)
#             # Convert date columns for uploaded data
#             date_columns = ['order_date', 'requested_delivery_date', 'planned_delivery_date', 
#                            'actual_delivery_date', 'cancellation_date']
#             for col in date_columns:
#                 if col in df.columns:
#                     df[col] = pd.to_datetime(df[col], errors='coerce')
#             st.session_state.data_loaded = True
#             st.session_state.df = df  # Store in session state
#             # Reset systems when new data is loaded
#             st.session_state.ml_predictor = None
#             st.session_state.alert_system = None
#         else:
#             df = load_data()
#             if df is not None:
#                 st.session_state.data_loaded = True
#                 st.session_state.df = df  # Store in session state
        
#         if st.session_state.data_loaded and df is not None:
#             # Get latest available date
#             latest_date = get_latest_data_date(df)
#             latest_month_year = latest_date.strftime("%B %Y")
            
#             st.success(f"Data loaded: {len(df)} orders")
#             st.info(f"Date range: {df['order_date'].min().strftime('%Y-%m-%d') if 'order_date' in df.columns else 'N/A'} to {latest_date.strftime('%Y-%m-%d')}")
            
#             # Business overview
#             st.markdown('<h3 class="section-header">Business Overview</h3>', unsafe_allow_html=True)
            
#             unique_orders = df['order_id'].nunique() if 'order_id' in df.columns else len(df)
            
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.metric("Orders Placed", format_large_number(unique_orders))
#                 if 'order_value_usd' in df.columns:
#                     total_revenue = df[df['order_status'].isin(['Completed'])]['order_value_usd'].sum()
#                     st.metric("Revenue Earned", format_currency(total_revenue))
#             with col2:
#                 if 'order_status' in df.columns:
#                     completed = df[df['order_status'].isin(['Completed'])]['order_id'].nunique()
#                     st.metric("Fulfilled Orders", format_large_number(completed))
#                 if 'is_delayed' in df.columns:
#                     delayed = df[df['is_delayed'] == True]['order_id'].nunique()
#                     st.metric("Delayed Deliveries", format_large_number(delayed))
            
#             # Executive Dashboard Controls (moved above alerts)
#             st.markdown('<h3 class="section-header">Executive Controls</h3>', unsafe_allow_html=True)
            
#             if st.button("Executive Dashboard", use_container_width=True, key="dashboard_btn"):
#                 st.session_state.show_dashboard = True
            
#             # Initialize and display alert system
#             alert_system = initialize_alert_system(df)
#             if alert_system:
#                 display_alerts(alert_system)
            
#             # Show model status
#             show_model_status()
        
#         # Analysis Modes
#         st.markdown('<h3 class="section-header">Analytics Capabilities</h3>', unsafe_allow_html=True)
#         st.markdown("""
#         **Standard Business Analytics**
#         - Pre-built operational metrics
#         - Rapid response generation
#         - Standard reporting queries
        
#         **Advanced Intelligence Analysis**
#         - Dynamic query interpretation
#         - Statistical analysis capabilities
#         - Custom business insights
        
#         **AI-Driven Prioritization**
#         - Pre-calculated ML intelligence
#         - Instant response processing
#         - Production-optimized deployment
#         """)
        
#         # Add placeholder questions to sidebar as well
#         if st.session_state.data_loaded and df is not None:
#             latest_date = get_latest_data_date(df)
#             latest_month_year = latest_date.strftime("%B %Y")
            
#             st.markdown('<h3 class="section-header">Quick Queries</h3>', unsafe_allow_html=True)
            
#             # Sidebar placeholder questions (same as main interface)
#             sidebar_questions = [
#                 f"Executive summary for {latest_month_year}",
#                 # f"Revenue performance analysis for {latest_month_year}",
#                 "Top operational challenges analysis",
#                 f"Advanced AI prioritization for {latest_month_year}"
#             ]
            
#             for i, question in enumerate(sidebar_questions):
#                 if st.button(question, key=f"sidebar_q_{i}", use_container_width=True):
#                     st.session_state.messages.append({
#                         "role": "user", 
#                         "content": question,
#                         "is_standard": True
#                     })
#                     st.rerun()
    
#     # Main interface
#     if not st.session_state.data_loaded or df is None:
#         st.markdown('<div class="warning-box">', unsafe_allow_html=True)
#         st.warning("Please upload your enterprise dataset or ensure orders.csv is in the data/ folder.")
#         st.info("Expected columns: order_id, order_date, customer_id, product_id, quantity, order_value_usd, order_status, is_delayed, delay_reason, etc.")
#         st.markdown('</div>', unsafe_allow_html=True)
#         return
    
#     # Check if dashboard should be shown
#     if st.session_state.show_dashboard:
#         visualizer = Visualizer(df)
#         visualizer.create_comprehensive_dashboard()
#         if st.button("Return to Analytics", key="back_to_chat"):
#             st.session_state.show_dashboard = False
#             st.rerun()
#         return
    
#     # Initialize components
#     try:
#         analyzer = DataAnalyzer(df)
#         chat_handler = ChatHandler()
#         visualizer = Visualizer(df)
#     except Exception as e:
#         st.error(f"Error initializing components: {str(e)}")
#         st.error("Please check your AI API configuration and data format.")
#         return
    
#     # Main chat interface
#     st.markdown('<div class="chat-container">', unsafe_allow_html=True)
#     # st.markdown('<h2 class="section-header">Executive Business Intelligence Interface</h2>', unsafe_allow_html=True)
    
#     # Show placeholder questions when no messages (always show)
#     latest_date = get_latest_data_date(df)
#     latest_month_year = latest_date.strftime("%B %Y")
#     display_placeholder_questions(latest_month_year)
    
#     # Display chat messages
#     for i, message in enumerate(st.session_state.messages):
#         with st.chat_message(message["role"]):
#             if message["role"] == "user":
#                 is_standard = message.get("is_standard", False)
#                 is_ai_query = is_ai_prioritization_query(message["content"])
                
#                 if is_ai_query:
#                     st.markdown('<div class="ai-query">Advanced AI Analytics Query</div>', unsafe_allow_html=True)
#                 elif is_standard:
#                     st.markdown('<div class="standard-query">Standard Business Analytics Query</div>', unsafe_allow_html=True)
#                 else:
#                     st.markdown('<div class="custom-query">Custom Intelligence Query</div>', unsafe_allow_html=True)
            
#             st.markdown(message["content"])
            
#             # Add analytics button for assistant responses
#             if message["role"] == "assistant":
#                 if st.button("View Executive Dashboard", key=f"analytics_btn_{i}"):
#                     st.session_state.show_analytics_modal = True
#                     st.rerun()
    
#     # Process pending user message
#     if has_pending_user_message():
#         last_message = st.session_state.messages[-1]
        
#         with st.chat_message("assistant"):
#             is_standard = last_message.get("is_standard", False)
            
#             # Check if this is an AI prioritization query
#             is_ai_query = is_ai_prioritization_query(last_message["content"])
            
#             if is_ai_query:
#                 with st.spinner("Executing advanced analytics pipeline..."):
#                     st.info("Optimized processing: Utilizing pre-calculated intelligence for instantaneous response")
#                     response = process_user_message(
#                         last_message["content"], 
#                         df, 
#                         analyzer, 
#                         chat_handler, 
#                         is_standard=is_standard
#                     )
#             elif is_standard:
#                 with st.spinner("Processing business analytics request..."):
#                     st.info("Executing standard business analytics")
#                     response = process_user_message(
#                         last_message["content"], 
#                         df, 
#                         analyzer, 
#                         chat_handler, 
#                         is_standard=True
#                     )
#             else:
#                 with st.spinner("Generating custom intelligence analysis..."):
#                     st.info("Creating dynamic business insights")
#                     response = process_user_message(
#                         last_message["content"], 
#                         df, 
#                         analyzer, 
#                         chat_handler, 
#                         is_standard=False
#                     )
            
#             st.markdown(response)
#             st.session_state.messages.append({"role": "assistant", "content": response})
            
#             # Add analytics button for new response
#             if st.button("View Executive Dashboard", key="analytics_btn_new"):
#                 st.session_state.show_analytics_modal = True
#                 st.rerun()
    
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     # Chat input
#     if prompt := st.chat_input("Enter your executive business question for advanced analytics..."):
#         st.session_state.messages.append({
#             "role": "user", 
#             "content": prompt
#         })
#         st.rerun()
    
#     # Show analytics modal if requested
#     show_analytics_modal()

# if __name__ == "__main__":
#     main()

# import streamlit as st
# import pandas as pd
# import os
# from dotenv import load_dotenv
# from data_analyzer import DataAnalyzer
# from chat_handler import ChatHandler
# from visualization import Visualizer
# from alert_system import AlertSystem
# # Import the production-optimized ML predictor
# from ml_predictor import OrderPrioritizationAI
# import datetime
# import traceback

# # Load environment variables
# load_dotenv()

# # Page configuration
# st.set_page_config(
#     page_title="Enterprise Order Analytics Platform",
#     page_icon="📊",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Professional CSS styling for enterprise applications + React-style sidebar
# st.markdown("""
# <style>
#     .main > div {
#         padding-top: 5rem;
#     }
    
#     .stApp {
#         background-color: #F8F9FA;
#     }
    
#     .css-1d391kg {
#         background-color: #FFFFFF;
#     }
    
#     /* Remove unwanted white space/bars */
#     .block-container {
#         padding-top: 1rem;
#         padding-bottom: 0rem;
#         padding-left: 170px; /* Add space for React-style sidebar - SIDEBAR WIDTH SETTING */
#     }
    
#     /* Hide Streamlit style elements */
#     #MainMenu {visibility: hidden;}
#     .stDeployButton {display:none;}
#     footer {visibility: hidden;}
#     .stApp > header {visibility: hidden;}
    
#     /* Hide default Streamlit sidebar initially */
#     .css-1d391kg {
#         display: none;
#         margin-top: 80px; /* Start below the pink top bar */
#     }
    
#     /* Fix for random white bars */
#     div[data-testid="stToolbar"] {
#         visibility: hidden;
#         height: 0%;
#         position: fixed;
#     }
    
#     div[data-testid="stDecoration"] {
#         visibility: hidden;
#         height: 0%;
#         position: fixed;
#     }
    
#     div[data-testid="stStatusWidget"] {
#         visibility: hidden;
#         height: 0%;
#         position: fixed;
#     }
    
#     /* Pink top bar */
#     .pink-top-bar {
#         background-color: #ffc0cb57;
#         height: 80px;
#         width: 100%;
#         position: fixed;
#         top: 0;
#         left: 0;
#         z-index: 1000;
#         display: flex;
#         align-items: center;
#         justify-content: space-between;
#         padding: 0 2rem;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#     }
    
#     .top-bar-logos {
#         display: flex;
#         align-items: center;
#         gap: 1rem;
#     }
    
#     .top-bar-admin {
#         display: flex;
#         align-items: center;
#         gap: 0.5rem;
#         color: #2C3E50;
#         font-weight: 600;
#         font-size: 1.1rem;
#     }
    
#     /* React-style sidebar */
#     .react-sidebar {
#         background-color: #ffc0cb57;
#         width: 150px; /* SIDEBAR WIDTH SETTING */
#         height: 100vh;
#         padding: 12px;
#         padding-top: 100px; /* Space for top bar */
#         color: black;
#         font-family: "Segoe UI", sans-serif;
#         display: flex;
#         flex-direction: column;
#         align-items: center;
#         overflow: hidden;
#         position: fixed;
#         left: 0;
#         top: 0;
#         z-index: 999;
#         transition: transform 0.3s ease;
#     }
    
#     .react-sidebar.collapsed {
#         transform: translateX(-150px);
#     }
    
#     .react-sidebar ul {
#         list-style: none;
#         padding: 0;
#         margin: 0;
#         width: 100%;
#     }
    
#     .react-sidebar li {
#         display: flex;
#         flex-direction: column;
#         align-items: center;
#         padding: 16px 0;
#         cursor: pointer;
#         transition: background-color 0.2s ease;
#         border-radius: 8px;
#         margin-bottom: 4px;
#     }
    
#     .react-sidebar li:hover {
#         background-color: #c257ab27;
#     }
    
#     .react-sidebar .menu-icon {
#         width: 25px;
#         height: 25px;
#         margin-bottom: 4px;
#         vertical-align: middle;
#     }
    
#     .react-sidebar .label {
#         font-size: 12px;
#         text-align: center;
#         font-weight: 600;
#     }
    
#     /* Collapse toggle button - always visible and hoverable */
#     .sidebar-toggle {
#         position: fixed;
#         top: 90px;
#         left: 10px;
#         z-index: 1001;
#         background-color: #ffc0cb;
#         border: none;
#         border-radius: 50%;
#         width: 40px;
#         height: 40px;
#         cursor: pointer;
#         display: flex;
#         align-items: center;
#         justify-content: center;
#         font-size: 18px;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.2);
#         transition: all 0.3s ease;
#         opacity: 0.7;
#         color: #2C3E50;
#         font-weight: bold;
#     }
    
#     .sidebar-toggle:hover {
#         background-color: #ffb6c1;
#         transform: scale(1.1);
#         opacity: 1;
#         box-shadow: 0 4px 8px rgba(0,0,0,0.3);
#     }
    
#     /* Show Streamlit sidebar when React sidebar is collapsed */
#     .react-sidebar.collapsed ~ .css-1d391kg {
#         display: block !important;
#         margin-left: 0;
#     }
    
#     .stSidebar .stButton > button {
#         background-color: #2C3E50;
#         color: white;
#         border: none;
#         border-radius: 6px;
#         font-weight: 500;
#         margin-bottom: 8px;
#         padding: 0.6rem 1rem;
#         font-size: 0.85rem;
#         transition: all 0.3s ease;
#         width: 100%;
#         text-align: left;
#     }
    
#     .stSidebar .stButton > button:hover {
#         background-color: #1A252F;
#         color: white;
#         transform: translateY(-1px);
#         box-shadow: 0 3px 6px rgba(0,0,0,0.15);
#     }
    
#     .chat-container {
#         background-color: white;
#         border-radius: 8px;
#         padding: 1.5rem;
#         box-shadow: 0 2px 8px rgba(0,0,0,0.1);
#         margin-bottom: 1rem;
#         max-height: 80vh;
#         overflow-y: auto;
#     }
    
#     .main-header {
#         color: #2C3E50;
#         font-size: 2.5rem;
#         font-weight: 600;
#         margin-bottom: 0.5rem;
#         font-family: 'Segoe UI', sans-serif;
#     }
    
#     .sub-header {
#         color: #5D6D7E;
#         font-size: 1.1rem;
#         margin-bottom: 2rem;
#         font-family: 'Segoe UI', sans-serif;
#     }
    
#     .section-header {
#         color: #2C3E50;
#         font-size: 1.2rem;
#         font-weight: 600;
#         margin-bottom: 1rem;
#         font-family: 'Segoe UI', sans-serif;
#     }
    
#     .standard-query {
#         background-color: #E3F2FD;
#         border-left: 4px solid #1976D2;
#         padding: 0.5rem;
#         margin: 0.5rem 0;
#         border-radius: 4px;
#     }
    
#     .custom-query {
#         background-color: #F3E5F5;
#         border-left: 4px solid #7B1FA2;
#         padding: 0.5rem;
#         margin: 0.5rem 0;
#         border-radius: 4px;
#     }
    
#     .ai-query {
#         background-color: #E8F5E8;
#         border-left: 4px solid #27AE60;
#         padding: 0.5rem;
#         margin: 0.5rem 0;
#         border-radius: 4px;
#     }
    
#     .info-box {
#         background-color: #E8F4FD;
#         border: 1px solid #2E86AB;
#         border-radius: 8px;
#         padding: 1rem;
#         margin: 1rem 0;
#     }
    
#     .success-box {
#         background-color: #E8F5E8;
#         border: 1px solid #27AE60;
#         border-radius: 8px;
#         padding: 1rem;
#         margin: 1rem 0;
#     }
    
#     .warning-box {
#         background-color: #FFF3CD;
#         border: 1px solid #FFC107;
#         border-radius: 8px;
#         padding: 1rem;
#         margin: 1rem 0;
#     }
    
#     .alert-critical {
#         background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
#         color: white;
#         padding: 0.8rem;
#         border-radius: 6px;
#         margin: 0.3rem 0;
#         box-shadow: 0 2px 6px rgba(231, 76, 60, 0.3);
#         font-size: 0.85rem;
#         font-weight: 500;
#     }
    
#     .alert-warning {
#         background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
#         color: white;
#         padding: 0.8rem;
#         border-radius: 6px;
#         margin: 0.3rem 0;
#         box-shadow: 0 2px 6px rgba(243, 156, 18, 0.3);
#         font-size: 0.85rem;
#         font-weight: 500;
#     }
    
#     .alert-info {
#         background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
#         color: white;
#         padding: 0.8rem;
#         border-radius: 6px;
#         margin: 0.3rem 0;
#         box-shadow: 0 2px 6px rgba(52, 152, 219, 0.3);
#         font-size: 0.85rem;
#         font-weight: 500;
#     }
    
#     .model-status-optimized {
#         background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
#         color: white;
#         padding: 1rem;
#         border-radius: 8px;
#         margin: 0.8rem 0;
#         box-shadow: 0 4px 12px rgba(30, 60, 114, 0.3);
#         border-left: 5px solid #ffffff;
#     }
    
#     .model-status-ready {
#         background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
#         color: white;
#         padding: 1rem;
#         border-radius: 8px;
#         margin: 0.8rem 0;
#         box-shadow: 0 4px 12px rgba(39, 174, 96, 0.3);
#         border-left: 5px solid #ffffff;
#     }
    
#     .model-status-training {
#         background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
#         color: white;
#         padding: 1rem;
#         border-radius: 8px;
#         margin: 0.8rem 0;
#         box-shadow: 0 4px 12px rgba(243, 156, 18, 0.3);
#         border-left: 5px solid #ffffff;
#     }
    
#     .performance-indicator {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         padding: 0.4rem 0.8rem;
#         border-radius: 15px;
#         font-size: 0.75rem;
#         font-weight: 600;
#         margin: 0.2rem 0;
#         display: inline-block;
#         box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3);
#     }
    
#     .logo-container {
#         display: flex;
#         justify-content: space-between;
#         align-items: center;
#         margin-bottom: 2rem;
#         padding: 1rem 0.5rem;
#     }
    
#     .logo-image {
#         max-width: 120px;
#         max-height: 80px;
#         object-fit: contain;
#     }
    
#     .placeholder-questions {
#         background-color: #F8F9FA;
#         border-radius: 8px;
#         padding: 1rem;
#         margin: 1rem 0;
#         border: 1px solid #E8E9EA;
#     }
    
#     .placeholder-question-button {
#         background-color: #2C3E50;
#         color: white;
#         border: none;
#         border-radius: 6px;
#         padding: 0.8rem 1rem;
#         margin: 0.5rem 0;
#         width: 100%;
#         font-size: 0.9rem;
#         font-weight: 500;
#         cursor: pointer;
#         transition: all 0.3s ease;
#     }
    
#     .placeholder-question-button:hover {
#         background-color: #1A252F;
#         transform: translateY(-1px);
#         box-shadow: 0 3px 6px rgba(0,0,0,0.15);
#     }
    
#     /* Smaller text input box - fixed positioning */
#     .stChatInputContainer {
#         max-width: 70% !important;
#         margin: 0 auto !important;
#         position: relative !important;
#         z-index: 100 !important;
#     }
    
#     .stChatInput {
#         margin-bottom: 100px !important;
#     }
    
#     /* Ensure chat input doesn't overlap with sidebar */
#     div[data-testid="stChatInput"] {
#         margin-left: 170px !important;
#         max-width: calc(100% - 200px) !important;
#     }
    
#     /* Adjust content when sidebar is collapsed */
#     .content-collapsed {
#         padding-left: 20px !important;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if 'messages' not in st.session_state:
#     st.session_state.messages = []
# if 'data_loaded' not in st.session_state:
#     st.session_state.data_loaded = False
# if 'show_dashboard' not in st.session_state:
#     st.session_state.show_dashboard = False
# if 'current_analysis' not in st.session_state:
#     st.session_state.current_analysis = None
# if 'show_analytics_modal' not in st.session_state:
#     st.session_state.show_analytics_modal = False
# if 'ml_predictor' not in st.session_state:
#     st.session_state.ml_predictor = None
# if 'alert_system' not in st.session_state:
#     st.session_state.alert_system = None
# if 'sidebar_collapsed' not in st.session_state:
#     st.session_state.sidebar_collapsed = False

# def format_large_number(value):
#     """Format large numbers with K, M, B suffixes"""
#     if pd.isna(value) or value == 0:
#         return "0"
    
#     if abs(value) >= 1_000_000_000:
#         return f"{value/1_000_000_000:.1f}B"
#     elif abs(value) >= 1_000_000:
#         return f"{value/1_000_000:.1f}M"
#     elif abs(value) >= 1_000:
#         return f"{value/1_000:.1f}K"
#     else:
#         return f"{value:,.0f}"

# def format_currency(value):
#     """Format currency values with K, M, B suffixes"""
#     if pd.isna(value) or value == 0:
#         return "$0"
    
#     if abs(value) >= 1_000_000_000:
#         return f"${value/1_000_000_000:.1f}B"
#     elif abs(value) >= 1_000_000:
#         return f"${value/1_000_000:.1f}M"
#     elif abs(value) >= 1_000:
#         return f"${value/1_000:.1f}K"
#     else:
#         return f"${value:,.0f}"

# @st.cache_data
# def load_data():
#     """Load and cache the enterprise dataset"""
#     try:
#         possible_paths = [
#             'data/orders.csv',
#             'orders.csv',
#             'data/dataset.csv'
#         ]
        
#         df = None
#         for path in possible_paths:
#             if os.path.exists(path):
#                 df = pd.read_csv(path)
#                 break
        
#         if df is None:
#             st.error("Dataset not found. Please upload your CSV file.")
#             return None
            
#         # Convert date columns
#         date_columns = ['order_date', 'requested_delivery_date', 'planned_delivery_date', 
#                        'actual_delivery_date', 'cancellation_date']
        
#         for col in date_columns:
#             if col in df.columns:
#                 df[col] = pd.to_datetime(df[col], errors='coerce')
        
#         return df
#     except Exception as e:
#         st.error(f"Error loading data: {str(e)}")
#         return None

# def initialize_ml_predictor(df):
#     """Initialize production-optimized ML predictor with pre-calculated cache"""
#     try:
#         if st.session_state.ml_predictor is None:
#             with st.spinner("Initializing AI-driven analytics system (initial setup: 2-3 minutes)..."):
#                 st.session_state.ml_predictor = OrderPrioritizationAI(df)
#                 st.success("AI analytics system operational. Subsequent analysis will execute instantly.")
#         return st.session_state.ml_predictor
#     except Exception as e:
#         st.error(f"Error initializing ML predictor: {str(e)}")
#         return None

# def initialize_alert_system(df):
#     """Initialize enterprise alert monitoring system"""
#     try:
#         if st.session_state.alert_system is None:
#             st.session_state.alert_system = AlertSystem(df)
#         return st.session_state.alert_system
#     except Exception as e:
#         st.error(f"Error initializing alert system: {str(e)}")
#         return None

# def display_alerts(alert_system):
#     """Display enterprise alerts in sidebar with critical alerts first"""
#     st.markdown('<h3 class="section-header">Enterprise Alert Center</h3>', unsafe_allow_html=True)
    
#     alerts = alert_system.generate_alerts()
#     alert_summary = alert_system.get_alert_summary()
    
#     # Alert summary metrics
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Critical", alert_summary['critical'], delta=None)
#     with col2:
#         st.metric("Warning", alert_summary['warning'], delta=None)
#     with col3:
#         st.metric("Info", alert_summary['info'], delta=None)
    
#     # Display alerts
#     if len(alerts) == 0:
#         st.success("All systems operating within normal parameters")
#     else:
#         # Filter and display critical alerts first
#         critical_alerts = [alert for alert in alerts if alert['severity'] == 'critical']
#         non_critical_alerts = [alert for alert in alerts if alert['severity'] != 'critical']
        
#         # Show critical alerts first
#         for alert in critical_alerts[:5]:
#             severity_class = f"alert-{alert['severity']}"
#             alert_text = alert['message']
#             st.markdown(f'<div class="{severity_class}">{alert_text}</div>', unsafe_allow_html=True)
        
#         # If there are more alerts (warning/info), show expand option
#         if len(non_critical_alerts) > 0:
#             remaining_count = len(non_critical_alerts)
#             st.info(f"Showing critical alerts. {remaining_count} warning/info alerts available")
            
#             if st.button("View All Alerts", key="view_all_alerts"):
#                 with st.expander("Warning and Info Alerts", expanded=True):
#                     for alert in non_critical_alerts:
#                         severity_class = f"alert-{alert['severity']}"
#                         st.markdown(f'<div class="{severity_class}">{alert["message"]}</div>', unsafe_allow_html=True)

# def is_ai_prioritization_query(user_query):
#     """Check if query specifically requests AI prioritization"""
#     ai_prioritization_keywords = [
#         'ai prioritization', 'ai-powered prioritization', 'ml prioritization',
#         'advanced ai prioritization', 'predictive prioritization', 
#         'ai-driven prioritization', 'machine learning prioritization',
#         'strategic prioritization with intervention', 'ai priority analysis',
#         'ai powered prioritization', 'generate ai powered prioritization',
#         'ai prioritization analysis', 'ml priority', 'ai prioritized orders',
#         'show me ai prioritized orders', 'ai prioritized', 'prioritized orders',
#         'ai models', 'machine learning models', 'ml models'
#     ]
    
#     query_lower = user_query.lower()
#     return any(keyword in query_lower for keyword in ai_prioritization_keywords)

# def display_placeholder_questions(latest_month_year):
#     """Display placeholder questions in main interface"""
#     st.markdown("### Get started with these common queries:")
    
#     # Placeholder questions
#     placeholder_questions = [
#         f"Executive summary for {latest_month_year}",
#         # f"Revenue performance analysis for {latest_month_year}",
#         "Top operational challenges analysis",
#         f"Advanced AI prioritization for {latest_month_year}"
#     ]
    
#     # Create buttons for each question
#     for i, question in enumerate(placeholder_questions):
#         if st.button(question, key=f"placeholder_q_{i}", use_container_width=True):
#             st.session_state.messages.append({
#                 "role": "user", 
#                 "content": question,
#                 "is_standard": True
#             })
#             st.rerun()

# def process_user_message(user_query, df, analyzer, chat_handler, is_standard=False):
#     """Process user message and generate intelligent response"""
#     try:
#         print(f"Processing message: '{user_query}' (standard: {is_standard})")
        
#         # For standard queries (button clicks), use the structured analysis
#         if is_standard:
#             # Analyze query requirements
#             analysis_result = analyzer.analyze_query(user_query)
            
#             # Check if this requires ML processing
#             if analysis_result.get("type") == "ai_prioritization_request":
#                 # Use the production-optimized ML predictor
#                 ml_predictor = initialize_ml_predictor(df)
#                 if ml_predictor:
#                     with st.spinner("Executing advanced analytics pipeline..."):
#                         st.info("Retrieving pre-calculated intelligence for instantaneous response")
#                         ai_results = ml_predictor.run_full_prioritization(current_month_only=True, top_n=20)
                        
#                         # Store results for analytics
#                         st.session_state.current_analysis = {
#                             'query': user_query,
#                             'result': ai_results,
#                             'is_standard': is_standard,
#                             'type': 'ai_prioritization'
#                         }
                        
#                         # Format AI response
#                         if 'error' in ai_results:
#                             return f"Analytics Error: {ai_results['error']}"
#                         else:
#                             return format_ai_prioritization_response(ai_results)
#                 else:
#                     return "Unable to initialize AI analytics system. Please verify data integrity and try again."
            
#             # For other standard queries, use standard analyzer
#             print(f"Analysis result type: {analysis_result.get('type', 'unknown')}")
            
#             # Store current analysis for analytics
#             st.session_state.current_analysis = {
#                 'query': user_query,
#                 'result': analysis_result,
#                 'is_standard': is_standard
#             }
            
#             # Generate intelligent response
#             response = chat_handler.get_response(user_query, analysis_result, df, is_placeholder=is_standard)
#             print(f"Response generated successfully")
            
#             return response
        
#         # For typed queries, check if it's AI prioritization
#         elif is_ai_prioritization_query(user_query):
#             # Use the production-optimized ML predictor for AI prioritization
#             ml_predictor = initialize_ml_predictor(df)
#             if ml_predictor:
#                 with st.spinner("Executing advanced analytics pipeline..."):
#                     st.info("Retrieving pre-calculated intelligence for instantaneous response")
#                     ai_results = ml_predictor.run_full_prioritization(current_month_only=True, top_n=20)
                    
#                     # Store results for analytics
#                     st.session_state.current_analysis = {
#                         'query': user_query,
#                         'result': ai_results,
#                         'is_standard': False,
#                         'type': 'ai_prioritization'
#                     }
                    
#                     # Format AI response
#                     if 'error' in ai_results:
#                         return f"Analytics Error: {ai_results['error']}"
#                     else:
#                         return format_ai_prioritization_response(ai_results)
#             else:
#                 return "Unable to initialize AI analytics system. Please verify data integrity and try again."
        
#         # For all other typed queries, use dynamic pandas analysis through OpenAI
#         else:
#             print(f"Processing typed query through OpenAI: {user_query}")
            
#             # Analyze query to get basic structure
#             analysis_result = analyzer.analyze_query(user_query)
            
#             # Store current analysis for analytics
#             st.session_state.current_analysis = {
#                 'query': user_query,
#                 'result': analysis_result,
#                 'is_standard': False,
#                 'type': 'dynamic'
#             }
            
#             # Generate response through OpenAI with pandas analysis
#             response = chat_handler.get_response(user_query, analysis_result, df, is_placeholder=False)
#             print(f"Dynamic response generated successfully")
            
#             return response
        
#     except Exception as e:
#         error_msg = f"Analysis encountered an error: {str(e)}"
#         print(f"Error processing message: {str(e)}")
#         st.error(error_msg)
#         st.error("Detailed error:")
#         st.code(traceback.format_exc())
#         return error_msg

# def format_ai_prioritization_response(ai_results):
#     """Format AI prioritization results into executive summary"""
#     try:
#         response = f"""
# # ADVANCED ANALYTICS: ORDER PRIORITIZATION GUIDANCE



# ## Strategically Prioritized Orders 
# """
        
#         # Add top 5 orders details
#         top_priorities = ai_results.get('top_priorities', pd.DataFrame())
#         if not top_priorities.empty:
#             for idx, row in top_priorities.head(10).iterrows():
#                 response += f"""
# ### Priority Rank #{idx+1}: Order {row.get('order_id', 'N/A')}
# - **Account**: {row.get('customer_name', 'N/A')} ({row.get('customer_tier', 'N/A')})
# - **Order Value**: ${row.get('order_value_usd', 0):,.0f}
# - **Priority Index**: {row.get('priority_score', 0):.1f}/100
# - **Risk Assessment**: {row.get('predicted_delay_probability', 0):.1%} probability
# - **Strategic Rationale**: {row.get('priority_reason', 'N/A')}
# """
        
#         # Add AI interventions - RESTORED FUNCTIONALITY
#         if ai_results.get('ai_interventions'):
#             response += "\n## AI-Recommended Interventions\n"
#             for i, intervention in enumerate(ai_results['ai_interventions'][:5]):
#                 response += f"\n### Order {intervention['order_id']} - {intervention['customer']}\n"
#                 response += f"- **Priority Score**: {intervention['priority_score']:.1f}/100\n"
#                 response += f"- **AI Delay Risk**: {intervention['ai_delay_risk']:.1%}\n"
#                 response += f"- **Predicted Delay**: {intervention['ai_predicted_days']:.0f} days\n"
#                 response += "\n**Recommended Actions**:\n"
#                 for action in intervention['interventions']:
#                     response += f"- {action}\n"
#                 response += "\n---\n"
        
#             response += f"""

# {ai_results.get('executive_summary', 'Analysis completed successfully')}

#  """
        
#         return response
        
#     except Exception as e:
#         return f"Advanced analytics completed successfully, but formatting encountered an issue: {str(e)}"

# def has_pending_user_message():
#     """Check if there's a user message that needs a response"""
#     if len(st.session_state.messages) == 0:
#         return False
    
#     last_message = st.session_state.messages[-1]
#     if last_message["role"] != "user":
#         return False
    
#     return len(st.session_state.messages) % 2 == 1

# def get_latest_data_date(df):
#     """Get the latest date available in the dataset"""
#     if 'order_date' in df.columns:
#         return df['order_date'].max()
#     return datetime.datetime(2025, 5, 31)  # Default to May 2025

# def show_model_status():
#     """Show AI model status and system controls"""
#     st.markdown('<h3 class="section-header">AI Analytics System Status</h3>', unsafe_allow_html=True)
    
#     if st.session_state.ml_predictor:
#         model_info = st.session_state.ml_predictor.get_model_status()
        
#         if model_info.get('models_loaded', False) and model_info.get('cache_loaded', False):
#             model_age = model_info.get('model_age_hours', 0)
#             cached_orders = model_info.get('cached_orders', 0)
            
#             st.markdown(f"""
#             <div class="model-status-optimized">
#                 <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">AI Analytics System: Operational</div>
#                 <div style="font-size: 1.4rem; font-weight: 700; margin: 0.5rem 0;">INSTANT PROCESSING ENABLED</div>
#                 <div style="font-size: 0.9rem; opacity: 0.9;">
#                     System Age: {model_age:.1f}h • Cache: {cached_orders:,} orders • Performance: {model_info.get('performance', 'Optimized')}
#                 </div>
#                 <div style="font-size: 0.8rem; margin-top: 0.5rem; opacity: 0.8;">
#                     Location: {model_info.get('models_folder', 'saved_models')} • Production Status: Operational
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)
#         elif model_info.get('models_loaded', False):
#             st.markdown(f"""
#             <div class="model-status-ready">
#                 <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">AI Models: Ready</div>
#                 <div style="font-size: 1.4rem; font-weight: 700; margin: 0.5rem 0;">CACHE INITIALIZATION...</div>
#                 <div style="font-size: 0.9rem; opacity: 0.9;">Next query will activate optimized processing</div>
#             </div>
#             """, unsafe_allow_html=True)
#         else:
#             st.markdown(f"""
#             <div class="model-status-training">
#                 <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">AI Models</div>
#                 <div style="font-size: 1.4rem; font-weight: 700; margin: 0.5rem 0;">READY FOR INITIALIZATION</div>
#                 <div style="font-size: 0.9rem; opacity: 0.9;">First query: 2-3 min setup, then optimized processing</div>
#             </div>
#             """, unsafe_allow_html=True)
#     else:
#         st.markdown(f"""
#         <div class="model-status-training">
#             <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">AI Analytics System</div>
#             <div style="font-size: 1.4rem; font-weight: 700; margin: 0.5rem 0;">NOT INITIALIZED</div>
#             <div style="font-size: 0.9rem; opacity: 0.9;">Will activate on first AI query</div>
#         </div>
#         """, unsafe_allow_html=True)
    
#     # System controls
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("Retrain Models", help="Retrain all AI models and rebuild cache"):
#             if st.session_state.ml_predictor:
#                 with st.spinner("Retraining models and rebuilding cache..."):
#                     st.session_state.ml_predictor.force_retrain_models()
#                 st.success("Models retrained and cache rebuilt successfully")
#                 st.rerun()
#             else:
#                 st.warning("Initialize ML predictor first by running an AI query")
    
#     with col2:
#         if st.button("System Details", help="Show detailed model and cache information"):
#             if st.session_state.ml_predictor:
#                 model_info = st.session_state.ml_predictor.get_model_status()
#                 st.json(model_info)
#             else:
#                 st.info("No ML predictor initialized yet")

# def show_analytics_modal():
#     """Display analytics in executive dashboard format"""
#     if st.session_state.show_analytics_modal and st.session_state.current_analysis:
        
#         with st.container():
#             st.markdown("---")
#             st.markdown("### Executive Analytics Dashboard")
            
#             # Close button
#             col1, col2 = st.columns([6, 1])
#             with col2:
#                 if st.button("Close", key="close_analytics"):
#                     st.session_state.show_analytics_modal = False
#                     st.rerun()
            
#             with col1:
#                 query = st.session_state.current_analysis['query']
#                 analysis_result = st.session_state.current_analysis['result']
#                 is_standard = st.session_state.current_analysis['is_standard']
#                 analysis_type = st.session_state.current_analysis.get('type', 'standard')
                
#                 # Show query type
#                 if analysis_type == 'ai_prioritization':
#                     st.success("Advanced AI Prioritization Analytics")
#                 elif is_standard:
#                     st.success("Standard Business Analytics")
#                 else:
#                     st.info("Custom Intelligence Analysis")
                
#                 st.markdown(f"**Executive Query:** {query}")
                
#                 # Initialize visualizer with current data
#                 if 'df' in st.session_state and st.session_state.df is not None:
#                     visualizer = Visualizer(st.session_state.df)
                    
#                     # Create enhanced visualizations
#                     try:
#                         visualizer.create_enhanced_visualizations(query, analysis_result)
                        
#                         # # Show AI prioritization summary if applicable
#                         # if analysis_type == 'ai_prioritization' or 'total_orders' in analysis_result:
#                         #     st.markdown("### AI Intelligence Summary")
                            
#                         #     col1, col2, col3, col4 = st.columns(4)
#                         #     with col1:
#                         #         st.metric(
#                         #             "High Priority Orders", 
#                         #             format_large_number(analysis_result.get('high_priority_count', 0))
#                         #         )
#                         #     with col2:
#                         #         st.metric(
#                         #             "Critical Risk Orders", 
#                         #             format_large_number(analysis_result.get('critical_risk_count', 0))
#                         #         )
#                         #     with col3:
#                         #         total_orders = analysis_result.get('total_orders', 1)
#                         #         risk_pct = (analysis_result.get('critical_risk_count', 0) / total_orders * 100) if total_orders > 0 else 0
#                         #         st.metric("Risk Percentage", f"{risk_pct:.1f}%")
#                         #     with col4:
#                         #         st.metric("Processing Time", analysis_result.get('processing_time', 'Optimized'))
                        
#                     except Exception as viz_error:
#                         st.error(f"Visualization error: {str(viz_error)}")
#                         st.info("Analysis completed successfully, but charts could not be generated.")

# def main():
#     # Create the logo HTML first
#     logo1_html = ""
#     logo2_html = ""
    
#     # Check for AI Foundry logo
#     try:
#         if os.path.exists("images/AI Foundry Logo.png"):
#             logo1_html = '<img src="data:image/png;base64,{}" style="height: 50px; margin-right: 15px;">'.format(
#                 __import__('base64').b64encode(open("images/AI Foundry Logo.png", "rb").read()).decode()
#             )
#         else:
#             logo1_html = '<div style="font-size: 1.3rem; font-weight: bold; color: #2C3E50; margin-right: 15px;">AI FOUNDRY</div>'
#     except:
#         logo1_html = '<div style="font-size: 1.3rem; font-weight: bold; color: #2C3E50; margin-right: 15px;">AI FOUNDRY</div>'
    
#     # Check for International logo  
#     try:
#         if os.path.exists("images/international.png"):
#             logo2_html = '<img src="data:image/png;base64,{}" style="height: 100px;">'.format(
#                 __import__('base64').b64encode(open("images/international.png", "rb").read()).decode()
#             )
#         else:
#             logo2_html = '<div style="font-size: 1.2rem; font-weight: bold; color: #2C3E50;">INTERNATIONAL</div>'
#     except:
#         logo2_html = '<div style="font-size: 1.2rem; font-weight: bold; color: #2C3E50;">INTERNATIONAL</div>'
    
#     # Pink top bar with logos and admin
#     st.markdown(f"""
#     <div class="pink-top-bar">
#         <div class="top-bar-logos">
#             {logo1_html}
#             {logo2_html}
#         </div>
#         <div class="top-bar-admin">
#             <div style="width: 35px; height: 35px; background-color: #ddd; border-radius: 50%; display: flex; align-items: center; justify-content: center;">👤</div>
#             <div>
#                 <div style="font-weight: 600;">Admin</div>
#                 <div style="font-size: 0.9rem; color: #666;">System Administrator</div>
#             </div>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Add always-visible toggle button with HTML
#     toggle_symbol = "☰" if not st.session_state.sidebar_collapsed else "←"
#     st.markdown(f"""
#     <button onclick="window.dispatchEvent(new CustomEvent('toggleSidebar'))" class="sidebar-toggle">
#         {toggle_symbol}
#     </button>
    
#     <script>
#     window.addEventListener('toggleSidebar', function() {{
#         // Trigger the hidden Streamlit button
#         const hiddenButton = window.parent.document.querySelector('[data-testid="baseButton-secondary"]:last-of-type');
#         if (hiddenButton) hiddenButton.click();
#     }});
#     </script>
#     """, unsafe_allow_html=True)
    
#     # Hidden Streamlit button for actual functionality
#     if st.button("", key="hidden_sidebar_toggle", help="", type="secondary"):
#         st.session_state.sidebar_collapsed = not st.session_state.sidebar_collapsed
#         st.rerun()
    
#     # Hide the hidden button
#     st.markdown("""
#     <style>
#     [data-testid="baseButton-secondary"]:last-of-type {
#         display: none !important;
#     }
#     </style>
#     """, unsafe_allow_html=True)
    
#     # React-style sidebar HTML
#     sidebar_class = "react-sidebar collapsed" if st.session_state.sidebar_collapsed else "react-sidebar"
#     st.markdown(f"""
#     <div class="{sidebar_class}">
#         <ul>
#             <li onclick="window.location.href='#dashboard'">
#                 <div class="menu-icon">🏠</div>
#                 <div class="label">Dashboard</div>
#             </li>
#             <li onclick="window.location.href='#agentic-qa'">
#                 <div class="menu-icon">💬</div>
#                 <div class="label">Agentic Q&A</div>
#             </li>
#             <li onclick="window.location.href='#smart-search'">
#                 <div class="menu-icon">🔍</div>
#                 <div class="label">Smart Search & Filtering</div>
#             </li>
#             <li onclick="window.location.href='#stock-prediction'">
#                 <div class="menu-icon">📊</div>
#                 <div class="label">Stock Out Prediction</div>
#             </li>
#             <li onclick="window.location.href='#quick-qa'">
#                 <div class="menu-icon">📋</div>
#                 <div class="label">Quick Q&A</div>
#             </li>
#             <li onclick="window.location.href='#restocking'">
#                 <div class="menu-icon">👁️</div>
#                 <div class="label">Restocking Optimization</div>
#             </li>
#         </ul>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Show/hide Streamlit sidebar based on collapse state
#     if st.session_state.sidebar_collapsed:
#         st.markdown("""
#         <style>
#         .css-1d391kg {
#             display: block !important;
#             margin-top: 80px !important;
#             padding-top: 1rem !important;
#         }
#         .block-container {
#             padding-left: 20px !important;
#         }
        
#         /* Ensure sidebar content starts below top bar */
#         .css-1d391kg > div {
#             margin-top: 0 !important;
#             padding-top: 0 !important;
#         }
#         </style>
#         """, unsafe_allow_html=True)
    
#     # Store df in session state for analytics
#     df = None
    
#     # Header
#     st.markdown('<h1 class="main-header">Enterprise Order Management Platform</h1>', unsafe_allow_html=True)
#     st.markdown('<p class="sub-header">Advanced AI-driven business intelligence with optimization support​</p>', unsafe_allow_html=True)
    
#     # Sidebar (only visible when React sidebar is collapsed)
#     if st.session_state.sidebar_collapsed:
#         with st.sidebar:
#             st.markdown('<h2 class="section-header">Data Configuration</h2>', unsafe_allow_html=True)
            
#             # File upload option
#             uploaded_file = st.file_uploader("Upload Enterprise Data", type=['csv'])
            
#             if uploaded_file is not None:
#                 df = pd.read_csv(uploaded_file)
#                 # Convert date columns for uploaded data
#                 date_columns = ['order_date', 'requested_delivery_date', 'planned_delivery_date', 
#                                'actual_delivery_date', 'cancellation_date']
#                 for col in date_columns:
#                     if col in df.columns:
#                         df[col] = pd.to_datetime(df[col], errors='coerce')
#                 st.session_state.data_loaded = True
#                 st.session_state.df = df  # Store in session state
#                 # Reset systems when new data is loaded
#                 st.session_state.ml_predictor = None
#                 st.session_state.alert_system = None
#             else:
#                 df = load_data()
#                 if df is not None:
#                     st.session_state.data_loaded = True
#                     st.session_state.df = df  # Store in session state
            
#             if st.session_state.data_loaded and df is not None:
#                 # Get latest available date
#                 latest_date = get_latest_data_date(df)
#                 latest_month_year = latest_date.strftime("%B %Y")
                
#                 st.success(f"Data loaded: {len(df)} orders")
#                 st.info(f"Date range: {df['order_date'].min().strftime('%Y-%m-%d') if 'order_date' in df.columns else 'N/A'} to {latest_date.strftime('%Y-%m-%d')}")
                
#                 # Business overview
#                 st.markdown('<h3 class="section-header">Business Overview</h3>', unsafe_allow_html=True)
                
#                 unique_orders = df['order_id'].nunique() if 'order_id' in df.columns else len(df)
                
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.metric("Orders Placed", format_large_number(unique_orders))
#                     if 'order_value_usd' in df.columns:
#                         total_revenue = df[df['order_status'].isin(['Completed'])]['order_value_usd'].sum()
#                         st.metric("Revenue Earned", format_currency(total_revenue))
#                 with col2:
#                     if 'order_status' in df.columns:
#                         completed = df[df['order_status'].isin(['Completed'])]['order_id'].nunique()
#                         st.metric("Fulfilled Orders", format_large_number(completed))
#                     if 'is_delayed' in df.columns:
#                         delayed = df[df['is_delayed'] == True]['order_id'].nunique()
#                         st.metric("Delayed Deliveries", format_large_number(delayed))
                
#                 # Executive Dashboard Controls (moved above alerts)
#                 st.markdown('<h3 class="section-header">Executive Controls</h3>', unsafe_allow_html=True)
                
#                 if st.button("Executive Dashboard", use_container_width=True, key="dashboard_btn"):
#                     st.session_state.show_dashboard = True
                
#                 # Initialize and display alert system
#                 alert_system = initialize_alert_system(df)
#                 if alert_system:
#                     display_alerts(alert_system)
                
#                 # Show model status
#                 show_model_status()
            
#             # Analysis Modes
#             st.markdown('<h3 class="section-header">Analytics Capabilities</h3>', unsafe_allow_html=True)
#             st.markdown("""
#             **Standard Business Analytics**
#             - Pre-built operational metrics
#             - Rapid response generation
#             - Standard reporting queries
            
#             **Advanced Intelligence Analysis**
#             - Dynamic query interpretation
#             - Statistical analysis capabilities
#             - Custom business insights
            
#             **AI-Driven Prioritization**
#             - Pre-calculated ML intelligence
#             - Instant response processing
#             - Production-optimized deployment
#             """)
            
#             # Add placeholder questions to sidebar as well
#             if st.session_state.data_loaded and df is not None:
#                 latest_date = get_latest_data_date(df)
#                 latest_month_year = latest_date.strftime("%B %Y")
                
#                 st.markdown('<h3 class="section-header">Quick Queries</h3>', unsafe_allow_html=True)
                
#                 # Sidebar placeholder questions (same as main interface)
#                 sidebar_questions = [
#                     f"Executive summary for {latest_month_year}",
#                     # f"Revenue performance analysis for {latest_month_year}",
#                     "Top operational challenges analysis",
#                     f"Advanced AI prioritization for {latest_month_year}"
#                 ]
                
#                 for i, question in enumerate(sidebar_questions):
#                     if st.button(question, key=f"sidebar_q_{i}", use_container_width=True):
#                         st.session_state.messages.append({
#                             "role": "user", 
#                             "content": question,
#                             "is_standard": True
#                         })
#                         st.rerun()
#     else:
#         # Load data when sidebar is not visible
#         df = load_data()
#         if df is not None:
#             st.session_state.data_loaded = True
#             st.session_state.df = df
    
#     # Main interface
#     if not st.session_state.data_loaded or df is None:
#         st.markdown('<div class="warning-box">', unsafe_allow_html=True)
#         st.warning("Please upload your enterprise dataset or ensure orders.csv is in the data/ folder.")
#         st.info("Expected columns: order_id, order_date, customer_id, product_id, quantity, order_value_usd, order_status, is_delayed, delay_reason, etc.")
#         st.markdown('</div>', unsafe_allow_html=True)
#         return
    
#     # Check if dashboard should be shown
#     if st.session_state.show_dashboard:
#         visualizer = Visualizer(df)
#         visualizer.create_comprehensive_dashboard()
#         if st.button("Return to Analytics", key="back_to_chat"):
#             st.session_state.show_dashboard = False
#             st.rerun()
#         return
    
#     # Initialize components
#     try:
#         analyzer = DataAnalyzer(df)
#         chat_handler = ChatHandler()
#         visualizer = Visualizer(df)
#     except Exception as e:
#         st.error(f"Error initializing components: {str(e)}")
#         st.error("Please check your AI API configuration and data format.")
#         return
    
#     # Main chat interface
#     st.markdown('<div class="chat-container">', unsafe_allow_html=True)
#     # st.markdown('<h2 class="section-header">Executive Business Intelligence Interface</h2>', unsafe_allow_html=True)
    
#     # Show placeholder questions when no messages (always show)
#     latest_date = get_latest_data_date(df)
#     latest_month_year = latest_date.strftime("%B %Y")
#     display_placeholder_questions(latest_month_year)
    
#     # Display chat messages
#     for i, message in enumerate(st.session_state.messages):
#         with st.chat_message(message["role"]):
#             if message["role"] == "user":
#                 is_standard = message.get("is_standard", False)
#                 is_ai_query = is_ai_prioritization_query(message["content"])
                
#                 if is_ai_query:
#                     st.markdown('<div class="ai-query">Advanced AI Analytics Query</div>', unsafe_allow_html=True)
#                 elif is_standard:
#                     st.markdown('<div class="standard-query">Standard Business Analytics Query</div>', unsafe_allow_html=True)
#                 else:
#                     st.markdown('<div class="custom-query">Custom Intelligence Query</div>', unsafe_allow_html=True)
            
#             st.markdown(message["content"])
            
#             # Add analytics button for assistant responses
#             if message["role"] == "assistant":
#                 if st.button("View Executive Dashboard", key=f"analytics_btn_{i}"):
#                     st.session_state.show_analytics_modal = True
#                     st.rerun()
    
#     # Process pending user message
#     if has_pending_user_message():
#         last_message = st.session_state.messages[-1]
        
#         with st.chat_message("assistant"):
#             is_standard = last_message.get("is_standard", False)
            
#             # Check if this is an AI prioritization query
#             is_ai_query = is_ai_prioritization_query(last_message["content"])
            
#             if is_ai_query:
#                 with st.spinner("Executing advanced analytics pipeline..."):
#                     st.info("Optimized processing: Utilizing pre-calculated intelligence for instantaneous response")
#                     response = process_user_message(
#                         last_message["content"], 
#                         df, 
#                         analyzer, 
#                         chat_handler, 
#                         is_standard=is_standard
#                     )
#             elif is_standard:
#                 with st.spinner("Processing business analytics request..."):
#                     st.info("Executing standard business analytics")
#                     response = process_user_message(
#                         last_message["content"], 
#                         df, 
#                         analyzer, 
#                         chat_handler, 
#                         is_standard=True
#                     )
#             else:
#                 with st.spinner("Generating custom intelligence analysis..."):
#                     st.info("Creating dynamic business insights")
#                     response = process_user_message(
#                         last_message["content"], 
#                         df, 
#                         analyzer, 
#                         chat_handler, 
#                         is_standard=False
#                     )
            
#             st.markdown(response)
#             st.session_state.messages.append({"role": "assistant", "content": response})
            
#             # Add analytics button for new response
#             if st.button("View Executive Dashboard", key="analytics_btn_new"):
#                 st.session_state.show_analytics_modal = True
#                 st.rerun()
    
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     # Chat input
#     if prompt := st.chat_input("Enter your executive business question for advanced analytics..."):
#         st.session_state.messages.append({
#             "role": "user", 
#             "content": prompt
#         })
#         st.rerun()
    
#     # Show analytics modal if requested
#     show_analytics_modal()

# if __name__ == "__main__":
#     main()


import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from data_analyzer import DataAnalyzer
from chat_handler import ChatHandler
from visualization import Visualizer
from alert_system import AlertSystem
# Import the production-optimized ML predictor
from ml_predictor import OrderPrioritizationAI
import datetime
import traceback

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Enterprise Order Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling for enterprise applications + React-style sidebar
st.markdown("""
<style>
    .main > div {
        padding-top: 5rem;
    }
    
    .stApp {
        background-color: #F8F9FA;
    }
    
    /* FORCE STREAMLIT SIDEBAR TO RIGHT SIDE */
    section[data-testid="stSidebar"] {
        position: fixed !important;
        right: 0 !important;
        left: auto !important;
        top: 80px !important;
        height: calc(100vh - 80px) !important;
        width: 300px !important;
        z-index: 998 !important;
        background-color: #FFFFFF !important;
        box-shadow: -2px 0 4px rgba(0,0,0,0.1) !important;
    }
    
    section[data-testid="stSidebar"] > div {
        background-color: #FFFFFF !important;
        padding-top: 1rem !important;
    }
    
    /* Remove unwanted white space/bars */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 170px; /* Space for left pink sidebar */
        padding-right: 320px; /* Space for right functional sidebar */
    }
    
    /* Hide Streamlit style elements */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    
    /* Hide sidebar close button */
    section[data-testid="stSidebar"] button[kind="header"] {
        display: none !important;
    }
    
    /* Fix for random white bars */
    div[data-testid="stToolbar"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
    
    div[data-testid="stDecoration"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
    
    div[data-testid="stStatusWidget"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
    
    /* Pink top bar */
    .pink-top-bar {
        background-color: #f5f5f5;
        height: 80px;
        width: 100%;
        position: fixed;
        top: 0;
        left: 0;
        z-index: 1000;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .top-bar-logos {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .top-bar-admin {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #2C3E50;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* PINK SIDEBAR (LEFT) - CUSTOM OVERLAY */
    .pink-sidebar {
        background-color: #f5f5f5;
        width: 150px;
        height: 100vh;
        padding: 12px;
        padding-top: 100px;
        color: black;
        font-family: "Segoe UI", sans-serif;
        display: flex;
        flex-direction: column;
        align-items: center;
        overflow: hidden;
        position: fixed;
        left: 0;
        top: 0;
        z-index: 999;
    }
    
    .pink-sidebar ul {
        list-style: none;
        padding: 0;
        margin: 0;
        width: 100%;
    }
    
    .pink-sidebar li {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 16px 0;
        cursor: pointer;
        transition: background-color 0.2s ease;
        border-radius: 8px;
        margin-bottom: 4px;
    }
    
    .pink-sidebar li:hover {
        background-color: #bbdefb;
    }
    
    .pink-sidebar .menu-icon {
        width: 25px;
        height: 25px;
        margin-bottom: 4px;
    }
    
    .pink-sidebar .label {
        font-size: 12px;
        text-align: center;
        font-weight: 600;
    }
    
    .stSidebar .stButton > button {
        background-color: #2C3E50;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 500;
        margin-bottom: 8px;
        padding: 0.6rem 1rem;
        font-size: 0.85rem;
        transition: all 0.3s ease;
        width: 100%;
        text-align: left;
    }
    
    .stSidebar .stButton > button:hover {
        background-color: #1A252F;
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 3px 6px rgba(0,0,0,0.15);
    }
    
    .chat-container {
        background-color: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        max-height: 80vh;
        overflow-y: auto;
    }
    
    .main-header {
        color: #2C3E50;
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-family: 'Segoe UI', sans-serif;
    }
    
    .sub-header {
        color: #5D6D7E;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-family: 'Segoe UI', sans-serif;
    }
    
    .section-header {
        color: #2C3E50;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        font-family: 'Segoe UI', sans-serif;
    }
    
    .standard-query {
        background-color: #E3F2FD;
        border-left: 4px solid #1976D2;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    
    .custom-query {
        background-color: #F3E5F5;
        border-left: 4px solid #7B1FA2;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    
    .ai-query {
        background-color: #E8F5E8;
        border-left: 4px solid #27AE60;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    
    .info-box {
        background-color: #E8F4FD;
        border: 1px solid #2E86AB;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #E8F5E8;
        border: 1px solid #27AE60;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #FFF3CD;
        border: 1px solid #FFC107;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .alert-critical {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        box-shadow: 0 2px 6px rgba(231, 76, 60, 0.3);
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        box-shadow: 0 2px 6px rgba(243, 156, 18, 0.3);
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .alert-info {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        box-shadow: 0 2px 6px rgba(52, 152, 219, 0.3);
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .model-status-optimized {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        box-shadow: 0 4px 12px rgba(30, 60, 114, 0.3);
        border-left: 5px solid #ffffff;
    }
    
    .model-status-ready {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        box-shadow: 0 4px 12px rgba(39, 174, 96, 0.3);
        border-left: 5px solid #ffffff;
    }
    
    .model-status-training {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        box-shadow: 0 4px 12px rgba(243, 156, 18, 0.3);
        border-left: 5px solid #ffffff;
    }
    
    .performance-indicator {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.2rem 0;
        display: inline-block;
        box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3);
    }
    
    /* Chat input positioning */
    div[data-testid="stChatInput"] {
        margin-left: 170px !important;
        margin-right: 320px !important;
        max-width: calc(100% - 520px) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'show_dashboard' not in st.session_state:
    st.session_state.show_dashboard = False
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'show_analytics_modal' not in st.session_state:
    st.session_state.show_analytics_modal = False
if 'ml_predictor' not in st.session_state:
    st.session_state.ml_predictor = None
if 'alert_system' not in st.session_state:
    st.session_state.alert_system = None
if 'sidebar_collapsed' not in st.session_state:
    st.session_state.sidebar_collapsed = False

def format_large_number(value):
    """Format large numbers with K, M, B suffixes"""
    if pd.isna(value) or value == 0:
        return "0"
    
    if abs(value) >= 1_000_000_000:
        return f"{value/1_000_000_000:.1f}B"
    elif abs(value) >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"{value/1_000:.1f}K"
    else:
        return f"{value:,.0f}"

def format_currency(value):
    """Format currency values with K, M, B suffixes"""
    if pd.isna(value) or value == 0:
        return "$0"
    
    if abs(value) >= 1_000_000_000:
        return f"${value/1_000_000_000:.1f}B"
    elif abs(value) >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"${value/1_000:.1f}K"
    else:
        return f"${value:,.0f}"

@st.cache_data
def load_data():
    """Load and cache the enterprise dataset"""
    try:
        possible_paths = [
            'data/orders.csv',
            'orders.csv',
            'data/dataset.csv'
        ]
        
        df = None
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                break
        
        if df is None:
            st.error("Dataset not found. Please upload your CSV file.")
            return None
            
        # Convert date columns
        date_columns = ['order_date', 'requested_delivery_date', 'planned_delivery_date', 
                       'actual_delivery_date', 'cancellation_date']
        
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def initialize_ml_predictor(df):
    """Initialize production-optimized ML predictor with pre-calculated cache"""
    try:
        if st.session_state.ml_predictor is None:
            with st.spinner("Initializing AI-driven analytics system (initial setup: 2-3 minutes)..."):
                st.session_state.ml_predictor = OrderPrioritizationAI(df)
                st.success("AI analytics system operational. Subsequent analysis will execute instantly.")
        return st.session_state.ml_predictor
    except Exception as e:
        st.error(f"Error initializing ML predictor: {str(e)}")
        return None

def initialize_alert_system(df):
    """Initialize enterprise alert monitoring system"""
    try:
        if st.session_state.alert_system is None:
            st.session_state.alert_system = AlertSystem(df)
        return st.session_state.alert_system
    except Exception as e:
        st.error(f"Error initializing alert system: {str(e)}")
        return None

def display_alerts(alert_system):
    """Display enterprise alerts in sidebar with critical alerts first"""
    st.markdown('<h3 class="section-header">Enterprise Alert Center</h3>', unsafe_allow_html=True)
    
    alerts = alert_system.generate_alerts()
    alert_summary = alert_system.get_alert_summary()
    
    # Alert summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Critical", alert_summary['critical'], delta=None)
    with col2:
        st.metric("Warning", alert_summary['warning'], delta=None)
    with col3:
        st.metric("Info", alert_summary['info'], delta=None)
    
    # Display alerts
    if len(alerts) == 0:
        st.success("All systems operating within normal parameters")
    else:
        # Filter and display critical alerts first
        critical_alerts = [alert for alert in alerts if alert['severity'] == 'critical']
        non_critical_alerts = [alert for alert in alerts if alert['severity'] != 'critical']
        
        # Show critical alerts first
        for alert in critical_alerts[:5]:
            severity_class = f"alert-{alert['severity']}"
            alert_text = alert['message']
            st.markdown(f'<div class="{severity_class}">{alert_text}</div>', unsafe_allow_html=True)
        
        # If there are more alerts (warning/info), show expand option
        if len(non_critical_alerts) > 0:
            remaining_count = len(non_critical_alerts)
            st.info(f"Showing critical alerts. {remaining_count} warning/info alerts available")
            
            if st.button("View All Alerts", key="view_all_alerts"):
                with st.expander("Warning and Info Alerts", expanded=True):
                    for alert in non_critical_alerts:
                        severity_class = f"alert-{alert['severity']}"
                        st.markdown(f'<div class="{severity_class}">{alert["message"]}</div>', unsafe_allow_html=True)

def is_ai_prioritization_query(user_query):
    """Check if query specifically requests AI prioritization"""
    ai_prioritization_keywords = [
        'ai prioritization', 'ai-powered prioritization', 'ml prioritization',
        'advanced ai prioritization', 'predictive prioritization', 
        'ai-driven prioritization', 'machine learning prioritization',
        'strategic prioritization with intervention', 'ai priority analysis',
        'ai powered prioritization', 'generate ai powered prioritization',
        'ai prioritization analysis', 'ml priority', 'ai prioritized orders',
        'show me ai prioritized orders', 'ai prioritized', 'prioritized orders',
        'ai models', 'machine learning models', 'ml models'
    ]
    
    query_lower = user_query.lower()
    return any(keyword in query_lower for keyword in ai_prioritization_keywords)

def display_placeholder_questions(latest_month_year):
    """Display placeholder questions in main interface"""
    st.markdown("### Get started with these common queries:")
    
    # Placeholder questions
    placeholder_questions = [
        f"Executive summary for {latest_month_year}",
        "Top operational challenges analysis",
        f"Advanced AI prioritization for {latest_month_year}"
    ]
    
    # Create buttons for each question
    for i, question in enumerate(placeholder_questions):
        if st.button(question, key=f"placeholder_q_{i}", use_container_width=True):
            st.session_state.messages.append({
                "role": "user", 
                "content": question,
                "is_standard": True
            })
            st.rerun()

# def process_user_message(user_query, df, analyzer, chat_handler, is_standard=False):
#     """Process user message and generate intelligent response"""
#     try:
#         print(f"Processing message: '{user_query}' (standard: {is_standard})")
        
#         # For standard queries (button clicks), use the structured analysis
#         if is_standard:
#             # Analyze query requirements
#             analysis_result = analyzer.analyze_query(user_query)
            
#             # Check if this requires ML processing
#             if analysis_result.get("type") == "ai_prioritization_request":
#                 # Use the production-optimized ML predictor
#                 ml_predictor = initialize_ml_predictor(df)
#                 if ml_predictor:
#                     with st.spinner("Executing advanced analytics pipeline..."):
#                         st.info("Retrieving pre-calculated intelligence for instantaneous response")
#                         ai_results = ml_predictor.run_full_prioritization(current_month_only=True, top_n=20)
                        
#                         # Store results for analytics
#                         st.session_state.current_analysis = {
#                             'query': user_query,
#                             'result': ai_results,
#                             'is_standard': is_standard,
#                             'type': 'ai_prioritization'
#                         }
                        
#                         # Format AI response
#                         if 'error' in ai_results:
#                             return f"Analytics Error: {ai_results['error']}"
#                         else:
#                             return format_ai_prioritization_response(ai_results)
#                 else:
#                     return "Unable to initialize AI analytics system. Please verify data integrity and try again."
            
#             # For other standard queries, use standard analyzer
#             print(f"Analysis result type: {analysis_result.get('type', 'unknown')}")
            
#             # Store current analysis for analytics
#             st.session_state.current_analysis = {
#                 'query': user_query,
#                 'result': analysis_result,
#                 'is_standard': is_standard
#             }
            
#             # Generate intelligent response
#             response = chat_handler.get_response(user_query, analysis_result, df, is_placeholder=is_standard)
#             print(f"Response generated successfully")
            
#             return response
        
#         # For typed queries, check if it's AI prioritization
#         elif is_ai_prioritization_query(user_query):
#             # Use the production-optimized ML predictor for AI prioritization
#             ml_predictor = initialize_ml_predictor(df)
#             if ml_predictor:
#                 with st.spinner("Executing advanced analytics pipeline..."):
#                     st.info("Retrieving pre-calculated intelligence for instantaneous response")
#                     ai_results = ml_predictor.run_full_prioritization(current_month_only=True, top_n=20)
                    
#                     # Store results for analytics
#                     st.session_state.current_analysis = {
#                         'query': user_query,
#                         'result': ai_results,
#                         'is_standard': False,
#                         'type': 'ai_prioritization'
#                     }
                    
#                     # Format AI response
#                     if 'error' in ai_results:
#                         return f"Analytics Error: {ai_results['error']}"
#                     else:
#                         return format_ai_prioritization_response(ai_results)
#             else:
#                 return "Unable to initialize AI analytics system. Please verify data integrity and try again."
        
#         # For all other typed queries, use dynamic pandas analysis through OpenAI
#         else:
#             print(f"Processing typed query through OpenAI: {user_query}")
            
#             # Analyze query to get basic structure
#             analysis_result = analyzer.analyze_query(user_query)
            
#             # Store current analysis for analytics
#             st.session_state.current_analysis = {
#                 'query': user_query,
#                 'result': analysis_result,
#                 'is_standard': False,
#                 'type': 'dynamic'
#             }
            
#             # Generate response through OpenAI with pandas analysis
#             response = chat_handler.get_response(user_query, analysis_result, df, is_placeholder=False)
#             print(f"Dynamic response generated successfully")
            
#             return response
        
#     except Exception as e:
#         error_msg = f"Analysis encountered an error: {str(e)}"
#         print(f"Error processing message: {str(e)}")
#         st.error(error_msg)
#         st.error("Detailed error:")
#         st.code(traceback.format_exc())
#         return error_msg
def process_user_message(user_query, df, analyzer, chat_handler, is_standard=False):
    """Process user message and generate intelligent response"""
    
    # Define cost info to append to all responses
    cost_info = "\n\n---\n**Query Cost Information:**\n\n"
    cost_info += "Input cost to run this query: $0.15 per 1M tokens\n\n"
    cost_info += "Output cost to run this query: $0.60 per 1M tokens"
    
    try:
        print(f"Processing message: '{user_query}' (standard: {is_standard})")
        
        # For standard queries (button clicks), use the structured analysis
        if is_standard:
            # Analyze query requirements
            analysis_result = analyzer.analyze_query(user_query)
            
            # Check if this requires ML processing
            if analysis_result.get("type") == "ai_prioritization_request":
                # Use the production-optimized ML predictor
                ml_predictor = initialize_ml_predictor(df)
                if ml_predictor:
                    with st.spinner("Executing advanced analytics pipeline..."):
                        st.info("Retrieving pre-calculated intelligence for instantaneous response")
                        ai_results = ml_predictor.run_full_prioritization(current_month_only=True, top_n=20)
                        
                        # Store results for analytics
                        st.session_state.current_analysis = {
                            'query': user_query,
                            'result': ai_results,
                            'is_standard': is_standard,
                            'type': 'ai_prioritization'
                        }
                        
                        # Format AI response
                        if 'error' in ai_results:
                            return f"Analytics Error: {ai_results['error']}" + cost_info
                        else:
                            return format_ai_prioritization_response(ai_results) + cost_info
                else:
                    return "Unable to initialize AI analytics system. Please verify data integrity and try again." + cost_info
            
            # For other standard queries, use standard analyzer
            print(f"Analysis result type: {analysis_result.get('type', 'unknown')}")
            
            # Store current analysis for analytics
            st.session_state.current_analysis = {
                'query': user_query,
                'result': analysis_result,
                'is_standard': is_standard
            }
            
            # Generate intelligent response
            response = chat_handler.get_response(user_query, analysis_result, df, is_placeholder=is_standard)
            print(f"Response generated successfully")
            
            return response + cost_info
        
        # For typed queries, check if it's AI prioritization
        elif is_ai_prioritization_query(user_query):
            # Use the production-optimized ML predictor for AI prioritization
            ml_predictor = initialize_ml_predictor(df)
            if ml_predictor:
                with st.spinner("Executing advanced analytics pipeline..."):
                    st.info("Retrieving pre-calculated intelligence for instantaneous response")
                    ai_results = ml_predictor.run_full_prioritization(current_month_only=True, top_n=20)
                    
                    # Store results for analytics
                    st.session_state.current_analysis = {
                        'query': user_query,
                        'result': ai_results,
                        'is_standard': False,
                        'type': 'ai_prioritization'
                    }
                    
                    # Format AI response
                    if 'error' in ai_results:
                        return f"Analytics Error: {ai_results['error']}" + cost_info
                    else:
                        return format_ai_prioritization_response(ai_results) + cost_info
            else:
                return "Unable to initialize AI analytics system. Please verify data integrity and try again." + cost_info
        
        # For all other typed queries, use dynamic pandas analysis through OpenAI
        else:
            print(f"Processing typed query through OpenAI: {user_query}")
            
            # Analyze query to get basic structure
            analysis_result = analyzer.analyze_query(user_query)
            
            # Store current analysis for analytics
            st.session_state.current_analysis = {
                'query': user_query,
                'result': analysis_result,
                'is_standard': False,
                'type': 'dynamic'
            }
            
            # Generate response through OpenAI with pandas analysis
            response = chat_handler.get_response(user_query, analysis_result, df, is_placeholder=False)
            print(f"Dynamic response generated successfully")
            
            return response + cost_info
        
    except Exception as e:
        error_msg = f"Analysis encountered an error: {str(e)}"
        print(f"Error processing message: {str(e)}")
        st.error(error_msg)
        st.error("Detailed error:")
        st.code(traceback.format_exc())
        return error_msg + cost_info
    
def format_ai_prioritization_response(ai_results):
    """Format AI prioritization results into executive summary"""
    try:
        response = f"""
# ADVANCED ANALYTICS: ORDER PRIORITIZATION GUIDANCE



## Strategically Prioritized Orders 
"""
        
        # Add top 5 orders details
        top_priorities = ai_results.get('top_priorities', pd.DataFrame())
        if not top_priorities.empty:
            for idx, row in top_priorities.head(10).iterrows():
                response += f"""
### Priority Rank #{idx+1}: Order {row.get('order_id', 'N/A')}
- **Account**: {row.get('customer_name', 'N/A')} ({row.get('customer_tier', 'N/A')})
- **Order Value**: ${row.get('order_value_usd', 0):,.0f}
- **Priority Index**: {row.get('priority_score', 0):.1f}/100
- **Risk Assessment**: {row.get('predicted_delay_probability', 0):.1%} probability
- **Strategic Rationale**: {row.get('priority_reason', 'N/A')}
"""
        
        # Add AI interventions - RESTORED FUNCTIONALITY
        if ai_results.get('ai_interventions'):
            response += "\n## AI-Recommended Interventions\n"
            for i, intervention in enumerate(ai_results['ai_interventions'][:5]):
                response += f"\n### Order {intervention['order_id']} - {intervention['customer']}\n"
                response += f"- **Priority Score**: {intervention['priority_score']:.1f}/100\n"
                response += f"- **AI Delay Risk**: {intervention['ai_delay_risk']:.1%}\n"
                response += f"- **Predicted Delay**: {intervention['ai_predicted_days']:.0f} days\n"
                response += "\n**Recommended Actions**:\n"
                for action in intervention['interventions']:
                    response += f"- {action}\n"
                response += "\n---\n"
        
            response += f"""

{ai_results.get('executive_summary', 'Analysis completed successfully')}

 """
        
        return response
        
    except Exception as e:
        return f"Advanced analytics completed successfully, but formatting encountered an issue: {str(e)}"

def has_pending_user_message():
    """Check if there's a user message that needs a response"""
    if len(st.session_state.messages) == 0:
        return False
    
    last_message = st.session_state.messages[-1]
    if last_message["role"] != "user":
        return False
    
    return len(st.session_state.messages) % 2 == 1

def get_latest_data_date(df):
    """Get the latest date available in the dataset"""
    if 'order_date' in df.columns:
        return df['order_date'].max()
    return datetime.datetime(2025, 5, 31)  # Default to May 2025

def show_model_status():
    """Show AI model status and system controls"""
    st.markdown('<h3 class="section-header">AI Analytics System Status</h3>', unsafe_allow_html=True)
    
    if st.session_state.ml_predictor:
        model_info = st.session_state.ml_predictor.get_model_status()
        
        if model_info.get('models_loaded', False) and model_info.get('cache_loaded', False):
            model_age = model_info.get('model_age_hours', 0)
            cached_orders = model_info.get('cached_orders', 0)
            
            st.markdown(f"""
            <div class="model-status-optimized">
                <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">AI Analytics System: Operational</div>
                <div style="font-size: 1.4rem; font-weight: 700; margin: 0.5rem 0;">INSTANT PROCESSING ENABLED</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">
                    System Age: {model_age:.1f}h • Cache: {cached_orders:,} orders • Performance: {model_info.get('performance', 'Optimized')}
                </div>
                <div style="font-size: 0.8rem; margin-top: 0.5rem; opacity: 0.8;">
                    Location: {model_info.get('models_folder', 'saved_models')} • Production Status: Operational
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif model_info.get('models_loaded', False):
            st.markdown(f"""
            <div class="model-status-ready">
                <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">AI Models: Ready</div>
                <div style="font-size: 1.4rem; font-weight: 700; margin: 0.5rem 0;">CACHE INITIALIZATION...</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Next query will activate optimized processing</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="model-status-training">
                <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">AI Models</div>
                <div style="font-size: 1.4rem; font-weight: 700; margin: 0.5rem 0;">READY FOR INITIALIZATION</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">First query: 2-3 min setup, then optimized processing</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="model-status-training">
            <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">AI Analytics System</div>
            <div style="font-size: 1.4rem; font-weight: 700; margin: 0.5rem 0;">NOT INITIALIZED</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">Will activate on first AI query</div>
        </div>
        """, unsafe_allow_html=True)
    
    # System controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Retrain Models", help="Retrain all AI models and rebuild cache"):
            if st.session_state.ml_predictor:
                with st.spinner("Retraining models and rebuilding cache..."):
                    st.session_state.ml_predictor.force_retrain_models()
                st.success("Models retrained and cache rebuilt successfully")
                st.rerun()
            else:
                st.warning("Initialize ML predictor first by running an AI query")
    
    with col2:
        if st.button("System Details", help="Show detailed model and cache information"):
            if st.session_state.ml_predictor:
                model_info = st.session_state.ml_predictor.get_model_status()
                st.json(model_info)
            else:
                st.info("No ML predictor initialized yet")

def show_analytics_modal():
    """Display analytics in executive dashboard format"""
    if st.session_state.show_analytics_modal and st.session_state.current_analysis:
        
        with st.container():
            st.markdown("---")
            st.markdown("### Executive Analytics Dashboard")
            
            # Close button
            col1, col2 = st.columns([6, 1])
            with col2:
                if st.button("Close", key="close_analytics"):
                    st.session_state.show_analytics_modal = False
                    st.rerun()
            
            with col1:
                query = st.session_state.current_analysis['query']
                analysis_result = st.session_state.current_analysis['result']
                is_standard = st.session_state.current_analysis['is_standard']
                analysis_type = st.session_state.current_analysis.get('type', 'standard')
                
                # Show query type
                if analysis_type == 'ai_prioritization':
                    st.success("Advanced AI Prioritization Analytics")
                elif is_standard:
                    st.success("Standard Business Analytics")
                else:
                    st.info("Custom Intelligence Analysis")
                
                st.markdown(f"**Executive Query:** {query}")
                
                # Initialize visualizer with current data
                if 'df' in st.session_state and st.session_state.df is not None:
                    visualizer = Visualizer(st.session_state.df)
                    
                    # Create enhanced visualizations
                    try:
                        visualizer.create_enhanced_visualizations(query, analysis_result)
                        
                    except Exception as viz_error:
                        st.error(f"Visualization error: {str(viz_error)}")
                        st.info("Analysis completed successfully, but charts could not be generated.")

def main():
    # Create the logo HTML first
    logo1_html = ""
    logo2_html = ""
    
    # Check for AI Foundry logo
    try:
        if os.path.exists("images/AI Foundry Logo.png"):
            logo1_html = '<img src="data:image/png;base64,{}" style="height: 50px; margin-right: 15px;">'.format(
                __import__('base64').b64encode(open("images/AI Foundry Logo.png", "rb").read()).decode()
            )
        else:
            logo1_html = '<div style="font-size: 1.3rem; font-weight: bold; color: #2C3E50; margin-right: 15px;">AI FOUNDRY</div>'
    except:
        logo1_html = '<div style="font-size: 1.3rem; font-weight: bold; color: #2C3E50; margin-right: 15px;">AI FOUNDRY</div>'
    
    # Check for International logo  
    try:
        if os.path.exists("images/international.png"):
            logo2_html = '<img src="data:image/png;base64,{}" style="height: 100px;">'.format(
                __import__('base64').b64encode(open("images/international.png", "rb").read()).decode()
            )
        else:
            logo2_html = '<div style="font-size: 1.2rem; font-weight: bold; color: #2C3E50;">INTERNATIONAL</div>'
    except:
        logo2_html = '<div style="font-size: 1.2rem; font-weight: bold; color: #2C3E50;">INTERNATIONAL</div>'
    
    # Pink top bar with logos and admin
    st.markdown(f"""
    <div class="pink-top-bar">
        <div class="top-bar-logos">
            {logo1_html}
            {logo2_html}
        </div>
        <div class="top-bar-admin">
            <div style="width: 35px; height: 35px; background-color: #ddd; border-radius: 50%; display: flex; align-items: center; justify-content: center;">👤</div>
            <div>
                <div style="font-weight: 600;">Admin</div>
                <div style="font-size: 0.9rem; color: #666;">System Administrator</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # PINK SIDEBAR (LEFT) - Custom HTML overlay
    st.markdown(f"""
    <div class="pink-sidebar">
        <ul>
            <li onclick="window.location.href='#dashboard'">
                <div class="menu-icon">🏠</div>
                <div class="label">Dashboard</div>
            </li>
            <li onclick="window.location.href='#upload-files'">
                <div class="menu-icon">📁</div>
                <div class="label">Upload Files</div>
            </li>
            <li onclick="window.location.href='#smart-search'">
                <div class="menu-icon">🔍</div>
                <div class="label">Connect to an Agent </div>
            </li>
            <li onclick="window.location.href='#stock-prediction'">
                <div class="menu-icon">📊</div>
                <div class="label">Stock Out Prediction</div>
            </li>
            <li onclick="window.location.href='#quick-qa'">
                <div class="menu-icon">📋</div>
                <div class="label">Quick Q&A</div>
            </li>
            <li onclick="window.location.href='#restocking'">
                <div class="menu-icon">👁️</div>
                <div class="label">Restocking Optimization</div>
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Store df in session state for analytics
    df = None
    
    # Header
    st.markdown('<h1 class="main-header">Enterprise Order Management Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced AI-driven business intelligence with optimization support​</p>', unsafe_allow_html=True)
    
    # RIGHT SIDEBAR - Streamlit sidebar forced to right with CSS
    with st.sidebar:
        st.markdown('<h2 class="section-header">Data Configuration</h2>', unsafe_allow_html=True)
        
        # File upload option
        uploaded_file = st.file_uploader("Upload Enterprise Data", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            # Convert date columns for uploaded data
            date_columns = ['order_date', 'requested_delivery_date', 'planned_delivery_date', 
                           'actual_delivery_date', 'cancellation_date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            st.session_state.data_loaded = True
            st.session_state.df = df  # Store in session state
            # Reset systems when new data is loaded
            st.session_state.ml_predictor = None
            st.session_state.alert_system = None
        else:
            df = load_data()
            if df is not None:
                st.session_state.data_loaded = True
                st.session_state.df = df  # Store in session state
        
        if st.session_state.data_loaded and df is not None:
            # Get latest available date
            latest_date = get_latest_data_date(df)
            latest_month_year = latest_date.strftime("%B %Y")
            
            st.success(f"Data loaded: {len(df)} orders")
            st.info(f"Date range: {df['order_date'].min().strftime('%Y-%m-%d') if 'order_date' in df.columns else 'N/A'} to {latest_date.strftime('%Y-%m-%d')}")
            
            # Business overview
            st.markdown('<h3 class="section-header">Business Overview</h3>', unsafe_allow_html=True)
            
            unique_orders = df['order_id'].nunique() if 'order_id' in df.columns else len(df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Orders Placed", format_large_number(unique_orders))
                if 'order_value_usd' in df.columns:
                    total_revenue = df[df['order_status'].isin(['Completed'])]['order_value_usd'].sum()
                    st.metric("Revenue Earned", format_currency(total_revenue))
            with col2:
                if 'order_status' in df.columns:
                    completed = df[df['order_status'].isin(['Completed'])]['order_id'].nunique()
                    st.metric("Fulfilled Orders", format_large_number(completed))
                if 'is_delayed' in df.columns:
                    delayed = df[df['is_delayed'] == True]['order_id'].nunique()
                    st.metric("Delayed Deliveries", format_large_number(delayed))
            
            # Executive Dashboard Controls (moved above alerts)
            st.markdown('<h3 class="section-header">Executive Controls</h3>', unsafe_allow_html=True)
            
            if st.button("Executive Dashboard", use_container_width=True, key="dashboard_btn"):
                st.session_state.show_dashboard = True
            
            # Initialize and display alert system
            alert_system = initialize_alert_system(df)
            if alert_system:
                display_alerts(alert_system)
            
            # Show model status
            show_model_status()
        
        # Analysis Modes
        st.markdown('<h3 class="section-header">Analytics Capabilities</h3>', unsafe_allow_html=True)
        st.markdown("""
        **Standard Business Analytics**
        - Pre-built operational metrics
        - Rapid response generation
        - Standard reporting queries
        
        **Advanced Intelligence Analysis**
        - Dynamic query interpretation
        - Statistical analysis capabilities
        - Custom business insights
        
        **AI-Driven Prioritization**
        - Pre-calculated ML intelligence
        - Instant response processing
        - Production-optimized deployment
        """)
        
        # Add placeholder questions to sidebar as well
        if st.session_state.data_loaded and df is not None:
            latest_date = get_latest_data_date(df)
            latest_month_year = latest_date.strftime("%B %Y")
            
            st.markdown('<h3 class="section-header">Quick Queries</h3>', unsafe_allow_html=True)
            
            # Sidebar placeholder questions (same as main interface)
            sidebar_questions = [
                f"Executive summary for {latest_month_year}",
                "Top operational challenges analysis",
                f"Advanced AI prioritization for {latest_month_year}"
            ]
            
            for i, question in enumerate(sidebar_questions):
                if st.button(question, key=f"sidebar_q_{i}", use_container_width=True):
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": question,
                        "is_standard": True
                    })
                    st.rerun()
    
    # Main interface
    if not st.session_state.data_loaded or df is None:
        st.warning("Please upload your enterprise dataset or ensure orders.csv is in the data/ folder.")
        st.info("Expected columns: order_id, order_date, customer_id, product_id, quantity, order_value_usd, order_status, is_delayed, delay_reason, etc.")
        return
    
    # Check if dashboard should be shown
    if st.session_state.show_dashboard:
        visualizer = Visualizer(df)
        visualizer.create_comprehensive_dashboard()
        if st.button("Return to Analytics", key="back_to_chat"):
            st.session_state.show_dashboard = False
            st.rerun()
        return
    
    # Initialize components
    try:
        analyzer = DataAnalyzer(df)
        chat_handler = ChatHandler()
        visualizer = Visualizer(df)
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        st.error("Please check your AI API configuration and data format.")
        return
    
    # Main chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Show placeholder questions when no messages (always show)
    latest_date = get_latest_data_date(df)
    latest_month_year = latest_date.strftime("%B %Y")
    display_placeholder_questions(latest_month_year)
    
    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                is_standard = message.get("is_standard", False)
                is_ai_query = is_ai_prioritization_query(message["content"])
                
                if is_ai_query:
                    st.markdown('<div class="ai-query">Advanced AI Analytics Query</div>', unsafe_allow_html=True)
                elif is_standard:
                    st.markdown('<div class="standard-query">Standard Business Analytics Query</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="custom-query">Custom Intelligence Query</div>', unsafe_allow_html=True)
            
            st.markdown(message["content"])
            
            # Add analytics button for assistant responses
            if message["role"] == "assistant":
                if st.button("View Executive Dashboard", key=f"analytics_btn_{i}"):
                    st.session_state.show_analytics_modal = True
                    st.rerun()
    
    # Process pending user message
    if has_pending_user_message():
        last_message = st.session_state.messages[-1]
        
        with st.chat_message("assistant"):
            is_standard = last_message.get("is_standard", False)
            
            # Check if this is an AI prioritization query
            is_ai_query = is_ai_prioritization_query(last_message["content"])
            
            if is_ai_query:
                with st.spinner("Executing advanced analytics pipeline..."):
                    st.info("Optimized processing: Utilizing pre-calculated intelligence for instantaneous response")
                    response = process_user_message(
                        last_message["content"], 
                        df, 
                        analyzer, 
                        chat_handler, 
                        is_standard=is_standard
                    )
            elif is_standard:
                with st.spinner("Processing business analytics request..."):
                    st.info("Executing standard business analytics")
                    response = process_user_message(
                        last_message["content"], 
                        df, 
                        analyzer, 
                        chat_handler, 
                        is_standard=True
                    )
            else:
                with st.spinner("Generating custom intelligence analysis..."):
                    st.info("Creating dynamic business insights")
                    response = process_user_message(
                        last_message["content"], 
                        df, 
                        analyzer, 
                        chat_handler, 
                        is_standard=False
                    )
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Add analytics button for new response
            if st.button("View Executive Dashboard", key="analytics_btn_new"):
                st.session_state.show_analytics_modal = True
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Enter your executive business question for advanced analytics..."):
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt
        })
        st.rerun()
    
    # Show analytics modal if requested
    show_analytics_modal()

if __name__ == "__main__":
    main()