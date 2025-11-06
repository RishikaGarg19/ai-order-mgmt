import openai
import os
import json
from datetime import datetime

class ChatHandler:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = "gpt-4o-mini"
        
        # Define standard questions for direct handling
        self.standard_questions = {
            "Show me orders for this month",
            "What's our revenue from completed orders this month?",
            "What are the top 5 delay challenges?",
            "How much revenue is impacted by delays?",
            "Show me orders by customer tier this month",
            "What's the average order value by product category?",
            "Executive summary for May 2025",
            "Revenue performance analysis for May 2025",
            "Top operational challenges analysis",
            "Revenue impact assessment for delays"
        }
        
        # AI prioritization keywords
        self.ai_prioritization_keywords = [
            'ai prioritization', 'ai-powered prioritization', 'ml prioritization',
            'advanced ai prioritization', 'predictive prioritization', 
            'ai-driven prioritization', 'machine learning prioritization',
            'strategic prioritization with intervention', 'ai priority analysis'
        ]
    
    def get_response(self, user_query, analysis_result, df, is_placeholder=False):
        """Generate AI response based on user query and data analysis"""
        
        # Check if this is a standard question OR if user typed exact standard text
        if is_placeholder or user_query.strip() in self.standard_questions:
            return self._handle_standard_response(user_query, analysis_result, df)
        
        # For typed queries, check if it's specifically requesting AI prioritization
        elif self._is_ai_prioritization_query(user_query):
            # This should trigger ML models
            if analysis_result.get("type") == "ai_prioritization_request":
                return self.handle_ai_prioritization(analysis_result)
            else:
                # Force ML processing by updating analysis result
                return "AI prioritization request detected. Please use the ML prioritization system."
        
        # For ALL other typed queries (non-standard, non-AI prioritization), use dynamic AI approach
        else:
            return self._handle_dynamic_typed_query(user_query, analysis_result, df)
    
    def _is_ai_prioritization_query(self, user_query):
        """Check if query is specifically requesting AI prioritization"""
        query_lower = user_query.lower()
        return any(keyword in query_lower for keyword in self.ai_prioritization_keywords)
    
    def _handle_standard_response(self, user_query, analysis_result, df):
        """Handle standard questions with direct responses"""
        
        # Handle AI prioritization
        if analysis_result.get("type") == "ai_prioritization_request":
            return self.handle_ai_prioritization(analysis_result)
        
        # Handle all other structured responses
        elif analysis_result.get("type") in [
            "current_month_orders", "revenue_analysis", "delay_challenges", 
            "revenue_impact", "customer_tier_analysis", "avg_order_value_analysis", "general_overview"
        ]:
            return self.handle_structured_analysis(user_query, analysis_result, df)
        
        # Handle errors
        elif "error" in analysis_result:
            return f"I encountered an issue: {analysis_result['error']}"
        
        else:
            return self.generate_general_response(user_query, analysis_result, df)
    
    def _handle_dynamic_typed_query(self, user_query, analysis_result, df):
        """Handle typed questions with dynamic AI pandas generation"""
        
        # Generate dynamic pandas code for ALL typed queries (except AI prioritization)
        return self.handle_dynamic_pandas_query(user_query, analysis_result, df)
    
    def handle_dynamic_pandas_query(self, user_query, analysis_result, df):
        """Generate and execute pandas queries dynamically using AI"""
        
        # Create prompt for pandas query generation
        pandas_prompt = self._create_pandas_prompt(user_query, df)
        
        try:
            print(f"Generating dynamic pandas code for: {user_query}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": pandas_prompt},
                    {"role": "user", "content": f"Generate pandas code for: {user_query}"}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            pandas_code = response.choices[0].message.content.strip()
            
            # Extract pandas code from response
            if "```python" in pandas_code:
                pandas_code = pandas_code.split("```python")[1].split("```")[0].strip()
            elif "```" in pandas_code:
                pandas_code = pandas_code.split("```")[1].strip()
            
            print(f"Generated pandas code:\n{pandas_code}")
            
            # Execute the pandas code
            from data_analyzer import DataAnalyzer
            analyzer = DataAnalyzer(df)
            result = analyzer.execute_pandas_query(pandas_code)
            
            print(f"Pandas execution result: {type(result)}")
            
            # Generate human-readable response using AI
            return self.generate_dynamic_response(user_query, pandas_code, result, df)
            
        except Exception as e:
            print(f"Error in dynamic pandas query: {str(e)}")
            return f"I encountered an error generating the dynamic analysis: {str(e)}"
    
    def generate_dynamic_response(self, user_query, pandas_code, result, df):
        """Generate human-readable response from pandas results using AI"""
        
        # Create response prompt with actual current date context
        current_date = datetime.now().strftime("%B %Y")
        current_month_year = f"{datetime.now().strftime('%B')} {datetime.now().year}"
        
        response_prompt = f"""
        You are an expert data analyst. The user asked: "{user_query}"
        
        IMPORTANT CONTEXT:
        - Current date: {current_date}
        - Current month: {current_month_year}
        - When user says "this month" or "current month", they mean {current_month_year}
        - This was a custom query, so provide detailed, insightful analysis
        
        The pandas analysis was executed and returned the following ACTUAL result:
        {str(result)[:3000]}
        
        
        Dataset context:
        - Total rows: {len(df)}
        - Date range: {df['order_date'].min() if 'order_date' in df.columns else 'N/A'} to {df['order_date'].max() if 'order_date' in df.columns else 'N/A'}
        - Available columns: {', '.join(df.columns.tolist()[:15])}...
        
        CRITICAL INSTRUCTIONS:
        1. Use ACTUAL numbers from the result - NO placeholders
        2. Provide DEEP INSIGHTS since this was a custom typed query
        3. If result shows specific values, report them exactly with proper formatting
        4. If result is empty or error, explain what might be wrong with the data
        5. When discussing "current month", always specify "{current_month_year}"
        6. Provide concrete, specific insights with real numbers
        7. Use professional language - no informal terms
        8. Focus on business insights, trends, and actionable recommendations
        9. Since this was typed, go deeper than standard responses
        10. Include comparative analysis where relevant (vs previous periods, benchmarks, etc.)
        11. If the question mentions revenue impact, use the column 'revenue_impact_usd'
        
        Generate a comprehensive business analysis response that:
        - Uses real numbers from the analysis result with proper formatting ($1.2M not $1,200,000)
        - Provides deeper insights than standard responses
        - Explains any data patterns or anomalies found
        - Offers strategic recommendations based on findings
        - Uses professional formatting with clear sections
        - Shows the executive calculation used instead of any pandas query
        """
        
        try:
            print("Generating detailed response with AI...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": response_prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Analysis completed, but I couldn't generate a summary: {str(e)}\n\nRaw result: {str(result)}\n\nPandas code used:\n```python\n{pandas_code}\n```"
    
    def handle_structured_analysis(self, user_query, analysis_result, df):
        """Handle structured analysis results with proper formatting"""
        
        analysis_type = analysis_result.get("type")
        
        if analysis_type == "current_month_orders":
            return self.format_current_month_response(analysis_result)
        elif analysis_type == "revenue_analysis":
            return self.format_revenue_response(analysis_result)
        elif analysis_type == "delay_challenges":
            return self.format_delay_challenges_response(analysis_result)
        elif analysis_type == "revenue_impact":
            return self.format_revenue_impact_response(analysis_result)
        elif analysis_type == "customer_tier_analysis":
            return self.format_customer_tier_response(analysis_result)
        elif analysis_type == "avg_order_value_analysis":
            return self.format_avg_order_value_response(analysis_result)
        elif analysis_type == "general_overview":
            return self.format_general_overview_response(analysis_result)
        else:
            return "Analysis completed, but I couldn't format the results properly."
    
    def handle_ai_prioritization(self, analysis_result):
        """Handle AI prioritization results with FULL intervention details"""
        
        if "error" in analysis_result:
            return f"AI Prioritization Error: {analysis_result['error']}"
        
        # Format the AI prioritization response
        response = f"""
# AI-Powered Order Prioritization Analysis

## Executive Summary
{analysis_result['executive_summary']}

## Key Metrics
- **Total Active Orders**: {analysis_result['total_orders']:,}
- **High Priority Orders**: {analysis_result['high_priority_count']:,}
- **Critical AI Risk Orders**: {analysis_result['critical_risk_count']:,}

## Top 5 Priority Orders
"""
        
        # Add top 5 orders details
        top_5 = analysis_result['top_priorities'].head(10)
        for idx, row in top_5.iterrows():
            response += f"""
### #{idx+1} Order {row['order_id']}
- **Customer**: {row.get('customer_name', 'N/A')} ({row.get('customer_tier', 'N/A')})
- **Value**: ${row.get('order_value_usd', 0):,.0f}
- **Priority Score**: {row['priority_score']:.1f}/100
- **AI Delay Risk**: {row['predicted_delay_probability']:.1%}
- **Reason**: {row['priority_reason']}
"""
        
        # Add AI interventions - RESTORED FUNCTIONALITY
        if analysis_result.get('ai_interventions'):
            response += "\n## AI-Recommended Interventions\n"
            for i, intervention in enumerate(analysis_result['ai_interventions'][:5]):
                response += f"\n### Order {intervention['order_id']} - {intervention['customer']}\n"
                response += f"- **Priority Score**: {intervention['priority_score']:.1f}/100\n"
                response += f"- **AI Delay Risk**: {intervention['ai_delay_risk']:.1%}\n"
                response += f"- **Predicted Delay**: {intervention['ai_predicted_days']:.0f} days\n"
                response += "\n**Recommended Actions**:\n"
                for action in intervention['interventions']:
                    response += f"- {action}\n"
                response += "\n---\n"
        
        # Add processing information
        response += f"""
## System Performance
- **Processing Time**: {analysis_result.get('processing_time', 'Optimized')}
- **Model Status**: {analysis_result.get('model_status', 'Ready')}
- **Cache Status**: Pre-calculated results for optimal performance

## Strategic Recommendations
1. **Immediate Action Required**: Focus on the top 5 priority orders
2. **Implement AI Interventions**: Execute the recommended actions above
3. **Monitor Critical Risk Orders**: Track the {analysis_result.get('critical_risk_count', 0)} high-risk orders daily
4. **Escalate as Needed**: Orders with >70% delay probability need executive attention

*Analysis leverages pre-computed ML intelligence for maximum operational efficiency.*
"""
        
        return response
    
    def format_current_month_response(self, result):
        """Format current month orders response"""
        month = result.get('month', 'Current Month')
        total_orders = result.get('total_orders', 0)
        total_value = result.get('total_value', 0)
        status_breakdown = result.get('status_breakdown', {})
        customer_breakdown = result.get('customer_breakdown', {})
        
        response = f"""
# {month} Orders Summary

## Key Metrics
- **Total Orders**: {total_orders:,}
- **Total Order Value**: ${total_value:,.2f}

## Order Status Breakdown
"""
#- **Average Order Value**: ${(total_value/total_orders):,.2f}
        
        for status, count in status_breakdown.items():
            percentage = (count / total_orders * 100) if total_orders > 0 else 0
            response += f"- **{status}**: {count:,} orders ({percentage:.1f}%)\n"
        
        if customer_breakdown:
            response += "\n## Customer Type Distribution\n"
            for customer_type, count in customer_breakdown.items():
                percentage = (count / total_orders * 100) if total_orders > 0 else 0
                response += f"- **{customer_type}**: {count:,} orders ({percentage:.1f}%)\n"
        
        response += f"\n## Business Insights\nYour business processed {total_orders:,} orders in {month}, generating ${total_value:,.2f} in total order value."
        
        return response
    
    def format_revenue_response(self, result):
        """Format revenue analysis response"""
        month = result.get('month', 'Current Month')
        total_revenue = result.get('total_revenue', 0)
        completed_orders = result.get('completed_orders', 0)
        avg_order_value = result.get('avg_order_value', 0)
        revenue_by_customer = result.get('revenue_by_customer', {})
        revenue_by_category = result.get('revenue_by_category', {})
        
        response = f"""
# Revenue Analysis - {month}

## Revenue Performance
- **Total Revenue from Completed Orders**: ${total_revenue:,.2f}
- **Number of Completed Orders**: {completed_orders:,}
- **Average Order Value**: ${avg_order_value:,.2f}

## Revenue by Customer Type
"""
        
        for customer_type, revenue in sorted(revenue_by_customer.items(), key=lambda x: x[1], reverse=True):
            percentage = (revenue / total_revenue * 100) if total_revenue > 0 else 0
            response += f"- **{customer_type}**: ${revenue:,.2f} ({percentage:.1f}%)\n"
        
        if revenue_by_category:
            response += "\n## Revenue by Product Category\n"
            for category, revenue in sorted(revenue_by_category.items(), key=lambda x: x[1], reverse=True):
                percentage = (revenue / total_revenue * 100) if total_revenue > 0 else 0
                response += f"- **{category}**: ${revenue:,.2f} ({percentage:.1f}%)\n"
        
        response += f"\n## Strategic Analysis\nCompleted orders in {month} generated ${total_revenue:,.2f} in revenue with an average order value of ${avg_order_value:,.2f}."
        
        return response
    
    def format_delay_challenges_response(self, result):
        """Format delay challenges response"""
        total_delayed = result.get('total_delayed', 0)
        delay_reasons = result.get('delay_reasons', {})
        secondary_delays = result.get('secondary_delays', {})
        severity_breakdown = result.get('severity_breakdown', {})
        avg_delay_days = result.get('avg_delay_days', 0)
        
        response = f"""
# Top 5 Delay Challenges Analysis

## Delay Overview
- **Orders that faced delays**: {total_delayed:,}
- **Average Delay Duration**: {avg_delay_days:.1f} days

### Primary Delay Categories
"""
        
        for i, (reason, count) in enumerate(delay_reasons.items(), 1):
            percentage = (count / total_delayed * 100) if total_delayed > 0 else 0
            response += f"{i}. **{reason}**: {count:,} orders ({percentage:.1f}%)\n"
        
        if secondary_delays:
            response += "\n### Secondary Delay Categories\n"
            for reason, count in list(secondary_delays.items())[:5]:
                percentage = (count / total_delayed * 100) if total_delayed > 0 else 0
                response += f"- **{reason}**: {count:,} orders ({percentage:.1f}%)\n"
        
        if severity_breakdown:
            response += "\n### Delay Impact Severity\n"
            for severity, count in severity_breakdown.items():
                percentage = (count / total_delayed * 100) if total_delayed > 0 else 0
                response += f"- **{severity}**: {count:,} orders ({percentage:.1f}%)\n"
        
        response += f"\n## Strategic Recommendations\nFocus on addressing the top delay categories, particularly the first one which accounts for the highest number of delayed orders."
        
        return response
    
    def format_revenue_impact_response(self, result):
        """Format revenue impact response"""
        total_impact = result.get('total_impact', 0)
        impact_by_reason = result.get('impact_by_reason', {})
        impact_by_severity = result.get('impact_by_severity', {})
        potential_lost = result.get('potential_lost', 0)
        penalty_cost = result.get('penalty_cost', 0)
        
        response = f"""
# Revenue Impact from Delays

## Financial Impact Summary
- **Total Revenue Impact**: ${total_impact:,.2f}
- **Potential Lost Revenue (Cancelled Orders)**: ${potential_lost:,.2f}
- **Late Delivery Penalties**: ${penalty_cost:,.2f}
- **Combined Impact**: ${(total_impact + potential_lost + penalty_cost):,.2f}

## Impact by Delay Category
"""
        
        for reason, impact in sorted(impact_by_reason.items(), key=lambda x: x[1], reverse=True):
            percentage = (impact / total_impact * 100) if total_impact > 0 else 0
            response += f"- **{reason}**: ${impact:,.2f} ({percentage:.1f}%)\n"
        
        if impact_by_severity:
            response += "\n## Impact by Severity Level\n"
            for severity, impact in sorted(impact_by_severity.items(), key=lambda x: x[1], reverse=True):
                percentage = (impact / total_impact * 100) if total_impact > 0 else 0
                response += f"- **{severity}**: ${impact:,.2f} ({percentage:.1f}%)\n"
        
        total_financial_impact = total_impact + potential_lost + penalty_cost
        response += f"\n## Business Impact Assessment\nDelays are costing your business ${total_financial_impact:,.2f} in direct revenue impact, lost sales, and penalties. Prioritize addressing the highest-impact delay categories."
        
        return response
    
    def format_customer_tier_response(self, result):
        """Format customer tier analysis response"""
        month = result.get('month', 'Current Month')
        tier_breakdown = result.get('tier_breakdown', {})
        revenue_by_tier = result.get('revenue_by_tier', {})
        avg_value_by_tier = result.get('avg_value_by_tier', {})
        total_orders = result.get('total_orders', 0)
        
        response = f"""
# Orders by Customer Tier - {month}

## Order Distribution by Tier
"""
        
        for tier, count in sorted(tier_breakdown.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_orders * 100) if total_orders > 0 else 0
            response += f"- **{tier}**: {count:,} orders ({percentage:.1f}%)\n"
        
        if revenue_by_tier:
            response += "\n## Revenue by Customer Tier\n"
            total_revenue = sum(revenue_by_tier.values())
            for tier, revenue in sorted(revenue_by_tier.items(), key=lambda x: x[1], reverse=True):
                percentage = (revenue / total_revenue * 100) if total_revenue > 0 else 0
                response += f"- **{tier}**: ${revenue:,.2f} ({percentage:.1f}%)\n"
        
        if avg_value_by_tier:
            response += "\n## Average Order Value by Tier\n"
            for tier, avg_value in sorted(avg_value_by_tier.items(), key=lambda x: x[1], reverse=True):
                response += f"- **{tier}**: ${avg_value:,.2f}\n"
        
        response += f"\n## Strategic Analysis\nAnalyze tier performance to optimize customer relationship strategies and identify upselling opportunities."
        
        return response
    
    def format_avg_order_value_response(self, result):
        """Format average order value analysis response"""
        avg_by_category = result.get('avg_by_category', {})
        overall_avg = result.get('overall_avg', 0)
        
        response = f"""
# Average Order Value by Product Category

## Overall Performance
- **Overall Average Order Value**: ${overall_avg:,.2f}

## Category Performance
"""
        
        # Sort by average order value
        sorted_categories = sorted(avg_by_category.items(), key=lambda x: x[1]['avg_order_value'], reverse=True)
        
        for category, data in sorted_categories:
            avg_value = data['avg_order_value']
            order_count = data['order_count']
            total_revenue = data['total_revenue']
            
            response += f"""
### {category}
- **Average Order Value**: ${avg_value:,.2f}
- **Total Orders**: {order_count:,}
- **Total Revenue**: ${total_revenue:,.2f}
- **vs Overall Avg**: {((avg_value/overall_avg - 1) * 100):+.1f}%
"""
        
        response += "\n## Strategic Recommendations\nFocus on high-value categories and consider strategies to increase order values in lower-performing categories."
        
        return response
    
    def format_general_overview_response(self, result):
        """Format general overview response"""
        stats = result.get('stats', {})
        
        response = f"""
# Dataset Overview

## Basic Statistics
- **Total Orders**: {stats.get('total_orders', 0):,}
- **Date Range**: {stats.get('date_range', 'N/A')}
- **Total Revenue**: ${stats.get('total_revenue', 0):,.2f}
- **Average Order Value**: ${stats.get('avg_order_value', 0):,.2f}

## Order Status Distribution
"""
        
        status_breakdown = stats.get('status_breakdown', {})
        total_orders = stats.get('total_orders', 1)
        
        for status, count in status_breakdown.items():
            percentage = (count / total_orders * 100) if total_orders > 0 else 0
            response += f"- **{status}**: {count:,} ({percentage:.1f}%)\n"
        
        customer_breakdown = stats.get('customer_type_breakdown', {})
        if customer_breakdown:
            response += "\n## Customer Type Distribution\n"
            for customer_type, count in customer_breakdown.items():
                percentage = (count / total_orders * 100) if total_orders > 0 else 0
                response += f"- **{customer_type}**: {count:,} ({percentage:.1f}%)\n"
        
        if 'delay_rate' in stats:
            response += f"\n## Operational Metrics\n"
            response += f"- **Delay Rate**: {stats['delay_rate']:.1f}%\n"
            response += f"- **Total Delayed Orders**: {stats.get('total_delayed', 0):,}\n"
        
        response += "\n## Executive Summary\nThis overview provides a comprehensive view of your order management performance and key operational metrics."
        
        return response
    
    def _create_pandas_prompt(self, user_query, df):
        """Create system prompt for pandas query generation"""
        
        current_date = datetime.now()
        current_date_str = current_date.strftime("%Y-%m-%d")
        current_month = current_date.month
        current_year = current_date.year
        current_month_name = current_date.strftime("%B")
        
        # Get actual column info from user's dataset
        actual_columns = df.columns.tolist()
        
        # Get date range info
        date_info = "N/A"
        if 'order_date' in df.columns:
            min_date = df['order_date'].min()
            max_date = df['order_date'].max()
            date_info = f"{min_date} to {max_date}"
        
        prompt = f"""
You are an expert pandas developer generating DYNAMIC code for business analytics queries.

CRITICAL CONTEXT:
- Current date: {current_date_str}
- Current month: {current_month} ({current_month_name})
- Current year: {current_year}
- When user says "this month" or "current month", they mean {current_month_name} {current_year}
- This is a TYPED QUERY, so be flexible and intelligent in interpretation

Dataset Information:
- DataFrame name: df
- Total rows: {len(df):,}
- Date range: {date_info}

AVAILABLE COLUMNS (EXACT NAMES):
{', '.join(actual_columns)}

KEY COLUMN MAPPINGS AND PRIORITIES:
- Orders: order_id, order_date, order_status
- Revenue Analysis: 
  * PRIMARY: revenue_impact_usd (USE THIS for revenue impact from delays and/or cancellations)
  * SECONDARY: order_value_usd (use for general order values)
  * TERTIARY: base_price_usd, profit_margin_percent
- Customers: customer_name, customer_type, customer_tier, customer_satisfaction
- Delays: is_delayed, delay_days, primary_delay_category, delay_reason
- Production: engineering_status, parts_availability, build_complexity
- Delivery: requested_delivery_date, actual_delivery_date, on_promise_delivery

CRITICAL COLUMN USAGE RULES:
1. For "revenue impact" queries → ALWAYS use 'revenue_impact_usd' instead of 'order_value_usd' 
2. For "total order value" or "order revenue" queries → use 'order_value_usd'
3. For delay analysis → use 'is_delayed' and 'delay_days'
4. For customer analysis → use customer_tier, customer_type
5. NEVER assume column names - use EXACT names from the available columns list

MANDATORY RULES:
1. ALWAYS assign your final result to a variable called 'result'
2. Be FLEXIBLE - interpret user intent, don't just match keywords
3. For current month queries, use: (df['order_date'].dt.month == {current_month}) & (df['order_date'].dt.year == {current_year})
4. Handle missing data with .fillna(0) or .dropna() temporarily
5. Use .sum(), .count(), .mean(), .median(), .std() for aggregations as needed
6. For top N results, use .head(N) or .nlargest(N)
7. Always check if columns exist before using them
8. Return meaningful data structures (dicts, DataFrames, or single values)
9. IMPORTANT: For order counts, use df['order_id'].nunique() to count unique orders
10. Be intelligent about date ranges, customer segments, product analysis, etc.

SPECIFIC QUERY PATTERNS:

"How much revenue is impacted because of delays?" →
```python
# Use revenue_impact_usd for delay impact analysis
if 'revenue_impact_usd' in df.columns:
    result = df[df['is_delayed']==True]['revenue_impact_usd'].sum()
else:
    # Fallback: calculate from delayed orders' order values
    delayed_orders = df[df['is_delayed'] == True]
    result = delayed_orders['order_value_usd'].sum()
```

"How much revenue is impacted because of cancellations?" →
```python
# Use revenue_impact_usd for delay impact analysis
if 'revenue_impact_usd' in df.columns:
    result = df[df['order_status']=='Canceled']['revenue_impact_usd'].sum()
else:
    # Fallback: calculate from canceled orders' order values
    canceled_orders = df[df['order_status'] == 'Canceled']
    result = canceled_orders['order_value_usd'].sum()
```

DYNAMIC INTERPRETATION EXAMPLES:

"Show me our biggest customers this quarter" ->
```python
current_quarter = df['order_date'].dt.quarter == {(current_month-1)//3 + 1}
current_year_mask = df['order_date'].dt.year == {current_year}
quarter_data = df[current_quarter & current_year_mask]
result = quarter_data.groupby('customer_name')['order_value_usd'].sum().nlargest(10).to_dict()
```

"What's the trend in delays over the last 6 months" ->
```python
six_months_ago = df['order_date'].max() - pd.DateOffset(months=6)
recent_data = df[df['order_date'] >= six_months_ago]
monthly_delays = recent_data.groupby(recent_data['order_date'].dt.to_period('M'))['is_delayed'].mean()
result = monthly_delays.to_dict()
```

"Compare performance between customer tiers" ->
```python
tier_performance = df.groupby('customer_tier').agg({{
    'order_value_usd': ['sum', 'mean', 'count'],
    'is_delayed': 'mean'
}}).round(2)
result = tier_performance.to_dict()
```

GENERATE INTELLIGENT PANDAS CODE that interprets the user's business question dynamically.
Don't just pattern match - understand the business intent and create appropriate analysis.

REMEMBER: 
- revenue_impact_usd = actual revenue lost due to delays
- order_value_usd = total value of orders
- Use the most appropriate column for each specific query type!
"""
        
        return prompt
    
    def generate_general_response(self, user_query, analysis_result, df):
        """Generate general response for fallback cases"""
        
        # Create system prompt
        system_prompt = self._create_system_prompt(df)
        
        # Create user prompt with analysis results
        user_prompt = self._create_user_prompt(user_query, analysis_result)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"I apologize, but I encountered an error generating the response: {str(e)}"
    
    def _create_system_prompt(self, df):
        """Create system prompt with dataset context"""
        current_date = datetime.now().strftime("%B %Y")
        
        # Dataset overview
        dataset_info = f"""
        Dataset Overview:
        - Total orders: {len(df)}
        - Date range: {df['order_date'].min().strftime('%Y-%m-%d') if 'order_date' in df.columns else 'N/A'} to {df['order_date'].max().strftime('%Y-%m-%d') if 'order_date' in df.columns else 'N/A'}
        - Current date: {current_date}
        - Available columns: {', '.join(df.columns.tolist())}
        """
        
        system_prompt = f"""
        You are an expert order analytics assistant for a manufacturing/automotive company. 
        You help analyze order data and provide insights about orders, revenue, delays, and prioritization.
        
        {dataset_info}
        
        Instructions:
        1. Provide clear, actionable insights based on the data analysis provided
        2. Use specific numbers and percentages when available
        3. Highlight key trends and patterns
        4. Suggest actionable recommendations when appropriate
        5. Be concise but comprehensive
        6. Format responses with proper structure using markdown
        7. When discussing revenue, always include currency symbols and proper formatting
        8. When mentioning dates, be specific about the time period being analyzed
        9. Use professional language
        10. Focus on business value and actionable insights
        
        Response Format:
        - Start with a brief summary/key takeaway
        - Provide detailed analysis with bullet points or numbered lists
        - Include relevant metrics and percentages
        - End with actionable recommendations if applicable
        
        Remember: You're analyzing real business data, so be professional and provide practical insights.
        """
        
        return system_prompt
    
    def _create_user_prompt(self, user_query, analysis_result):
        """Create user prompt with query and analysis results"""
        
        if "error" in analysis_result:
            return f"""
            User Query: {user_query}
            
            Error: {analysis_result['error']}
            
            Please provide a helpful response explaining what data might be missing and suggest alternative analysis approaches.
            """
        
        # Convert analysis result to readable format
        analysis_text = self._format_analysis_result(analysis_result)
        
        user_prompt = f"""
        User Query: {user_query}
        
        Data Analysis Results:
        {analysis_text}
        
        Please provide a comprehensive response to the user's query based on this analysis.
        Include specific insights, trends, and actionable recommendations.
        """
        
        return user_prompt
    
    def _format_analysis_result(self, result):
        """Format analysis result for the AI prompt"""
        
        analysis_type = result.get("type", "unknown")
        formatted_text = f"Analysis Type: {analysis_type}\n\n"
        
        # Add basic formatting for different result types
        if isinstance(result, dict):
            for key, value in result.items():
                if key != "type":
                    formatted_text += f"{key}: {value}\n"
        
        return formatted_text