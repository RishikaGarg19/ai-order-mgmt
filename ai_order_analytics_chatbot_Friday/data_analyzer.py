
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

class DataAnalyzer:
    def __init__(self, df):
        self.df = df
        # Fixed to use May 2025 as current month since that's the latest data available
        self.current_date = datetime(2025, 5, 31)  # Set to end of May 2025
        self.current_month = 5  # May
        self.current_year = 2025
        
        # NOTE: ML predictor is now handled separately in ml_predictor.py
        # This analyzer focuses on standard business analytics
    
    def analyze_query(self, query):
        """Analyze the user query and return relevant data insights"""
        query_lower = query.lower()
        
        # Check if this is an AI/ML prioritization query
        if any(keyword in query_lower for keyword in ['prioritization', 'priority', 'ai', 'ml', 'predict', 'forecast']):
            # Return a flag indicating this should use the ML predictor
            return {
                "type": "ai_prioritization_request",
                "message": "This query requires AI/ML processing. Use the ML predictor module.",
                "query": query
            }
        
        # Check for specific query patterns and route accordingly
        elif any(keyword in query_lower for keyword in ['may 2025', 'current month', 'orders for']):
            return self.get_current_month_orders()
        
        elif any(keyword in query_lower for keyword in ['revenue', 'accomplished', 'completed']):
            return self.get_revenue_analysis()
        
        elif any(keyword in query_lower for keyword in ['challenges', 'delay', 'top', 'delaying']):
            return self.get_delay_challenges()
        
        elif any(keyword in query_lower for keyword in ['revenue impact', 'impacted']):
            return self.get_revenue_impact()
        
        elif any(keyword in query_lower for keyword in ['customer tier', 'by customer']):
            return self.get_orders_by_customer_tier()
        
        elif any(keyword in query_lower for keyword in ['average order value', 'product category']):
            return self.get_avg_order_value_by_category()
        
        else:
            return self.get_general_overview()
    
    def execute_pandas_query(self, pandas_code):
        """Execute dynamically generated pandas code safely"""
        try:
            # Create a safe environment for pandas execution
            safe_globals = {
                'pd': pd,
                'np': np,
                'df': self.df,
                'current_date': self.current_date,
                'current_month': self.current_month,
                'current_year': self.current_year,
                'datetime': datetime,
                'timedelta': timedelta
            }
            
            # Execute the pandas code
            exec_locals = {}
            exec(pandas_code, safe_globals, exec_locals)
            
            # Return the result (look for 'result' variable or last expression)
            if 'result' in exec_locals:
                return exec_locals['result']
            else:
                # If no 'result' variable, try to evaluate the last line
                lines = pandas_code.strip().split('\n')
                last_line = lines[-1].strip()
                if last_line and not last_line.startswith('#'):
                    return eval(last_line, safe_globals)
                else:
                    return "Query executed successfully"
            
        except Exception as e:
            return f"Error executing query: {str(e)}"
    
    def get_current_month_orders(self):
        """Get orders for May 2025 using actual data"""
        if 'order_date' not in self.df.columns:
            return {"error": "order_date column not found"}
        
        # Filter for May 2025
        may_2025_mask = (
            (self.df['order_date'].dt.month == 5) & 
            (self.df['order_date'].dt.year == 2025)
        )
        may_orders = self.df[may_2025_mask]
        
        # Summary statistics
        total_orders = may_orders['order_id'].nunique() if 'order_id' in self.df.columns else len(may_orders)
        total_value = may_orders['order_value_usd'].sum() if 'order_value_usd' in self.df.columns else 0
        
        # Status breakdown
        status_breakdown = {}
        if 'order_status' in self.df.columns:
            status_breakdown = may_orders['order_status'].value_counts().to_dict()
        
        # Customer breakdown
        customer_breakdown = {}
        if 'customer_type' in self.df.columns:
            customer_breakdown = may_orders['customer_type'].value_counts().to_dict()
        
        return {
            "type": "current_month_orders",
            "total_orders": total_orders,
            "total_value": total_value,
            "status_breakdown": status_breakdown,
            "customer_breakdown": customer_breakdown,
            "data": may_orders,
            "month": "May 2025"
        }
    
    def get_revenue_analysis(self):
        """Analyze revenue from accomplished/completed orders for May 2025"""
        if 'order_status' not in self.df.columns or 'order_value_usd' not in self.df.columns:
            return {"error": "Required columns not found (order_status or order_value_usd)"}
        
        # Filter completed orders for May 2025
        may_2025_mask = (
            (self.df['order_date'].dt.month == 5) & 
            (self.df['order_date'].dt.year == 2025)
        )
        may_orders = self.df[may_2025_mask]
        
        # Completed orders - using actual order statuses from your dataset
        completed_statuses = ['Completed', 'Delivered', 'Shipped', 'Fulfilled']
        completed_orders = may_orders[
            may_orders['order_status'].isin(completed_statuses)
        ]
        
        total_revenue = completed_orders['order_value_usd'].sum()
        order_count = completed_orders['order_id'].nunique() if 'order_id' in self.df.columns else len(completed_orders)
        avg_order_value = total_revenue / order_count if order_count > 0 else 0
        
        # Revenue by customer type
        revenue_by_customer = {}
        if 'customer_type' in self.df.columns:
            revenue_by_customer = completed_orders.groupby('customer_type')['order_value_usd'].sum().to_dict()
        
        # Revenue by product category
        revenue_by_category = {}
        if 'product_category' in self.df.columns:
            revenue_by_category = completed_orders.groupby('product_category')['order_value_usd'].sum().to_dict()
        
        return {
            "type": "revenue_analysis",
            "total_revenue": total_revenue,
            "completed_orders": order_count,
            "avg_order_value": avg_order_value,
            "revenue_by_customer": revenue_by_customer,
            "revenue_by_category": revenue_by_category,
            "data": completed_orders,
            "month": "May 2025"
        }
    
    def get_delay_challenges(self):
        """Identify top challenges causing delays"""
        if 'is_delayed' not in self.df.columns:
            return {"error": "is_delayed column not found"}
        
        delayed_orders = self.df[self.df['is_delayed'] == True]
        
        if len(delayed_orders) == 0:
            return {
                "type": "delay_challenges",
                "total_delayed": 0,
                "delay_reasons": {},
                "secondary_delays": {},
                "severity_breakdown": {},
                "avg_delay_days": 0,
                "data": delayed_orders
            }
        
        # Top delay reasons - using actual column names from your dataset
        delay_reasons = {}
        if 'primary_delay_category' in self.df.columns:
            delay_reasons = delayed_orders['primary_delay_category'].value_counts().head(5).to_dict()
        elif 'delay_reason' in self.df.columns:
            delay_reasons = delayed_orders['delay_reason'].value_counts().head(5).to_dict()
        
        # Secondary delay categories
        secondary_delays = {}
        if 'secondary_delay_category' in self.df.columns:
            secondary_delays = delayed_orders['secondary_delay_category'].value_counts().head(5).to_dict()
        
        # Delay impact severity
        severity_breakdown = {}
        if 'delay_impact_severity' in self.df.columns:
            severity_breakdown = delayed_orders['delay_impact_severity'].value_counts().to_dict()
        
        # Average delay days
        avg_delay_days = 0
        if 'delay_days' in self.df.columns:
            avg_delay_days = delayed_orders['delay_days'].mean()
        
        return {
            "type": "delay_challenges",
            "total_delayed": len(delayed_orders),
            "delay_reasons": delay_reasons,
            "secondary_delays": secondary_delays,
            "severity_breakdown": severity_breakdown,
            "avg_delay_days": avg_delay_days,
            "data": delayed_orders
        }
    
    def get_revenue_impact(self):
        """Calculate revenue impact from delays and challenges"""
        if 'is_delayed' not in self.df.columns:
            return {"error": "is_delayed column not found"}
        
        delayed_orders = self.df[self.df['is_delayed'] == True]
        
        if len(delayed_orders) == 0:
            return {
                "type": "revenue_impact",
                "total_impact": 0,
                "impact_by_reason": {},
                "impact_by_severity": {},
                "potential_lost": 0,
                "penalty_cost": 0,
                "data": delayed_orders
            }
        
        # Total revenue impact
        total_impact = 0
        if 'revenue_impact_usd' in self.df.columns:
            total_impact = delayed_orders['revenue_impact_usd'].sum()
        else:
            # Calculate estimated impact based on order values and delay severity
            if 'order_value_usd' in self.df.columns and 'delay_days' in self.df.columns:
                # Estimate 1% revenue impact per day of delay
                total_impact = (delayed_orders['order_value_usd'] * delayed_orders['delay_days'] * 0.01).sum()
        
        # Impact by delay reason
        impact_by_reason = {}
        if 'primary_delay_category' in self.df.columns and 'revenue_impact_usd' in self.df.columns:
            impact_by_reason = delayed_orders.groupby('primary_delay_category')['revenue_impact_usd'].sum().to_dict()
        elif 'primary_delay_category' in self.df.columns and 'order_value_usd' in self.df.columns:
            # Estimate impact by category using order values
            impact_by_reason = delayed_orders.groupby('primary_delay_category')['order_value_usd'].sum().to_dict()
        
        # Impact by severity
        impact_by_severity = {}
        if 'delay_impact_severity' in self.df.columns and 'revenue_impact_usd' in self.df.columns:
            impact_by_severity = delayed_orders.groupby('delay_impact_severity')['revenue_impact_usd'].sum().to_dict()
        
        # Potential lost revenue (cancelled orders)
        cancelled_orders = self.df[self.df['order_status'] == 'Cancelled']
        potential_lost = cancelled_orders['order_value_usd'].sum() if 'order_value_usd' in self.df.columns else 0
        
        # Late delivery penalties
        penalty_cost = 0
        if 'late_delivery_penalty_usd' in self.df.columns:
            penalty_cost = delayed_orders['late_delivery_penalty_usd'].sum()
        
        return {
            "type": "revenue_impact",
            "total_impact": total_impact,
            "impact_by_reason": impact_by_reason,
            "impact_by_severity": impact_by_severity,
            "potential_lost": potential_lost,
            "penalty_cost": penalty_cost,
            "data": delayed_orders
        }
    
    def get_orders_by_customer_tier(self):
        """Get orders breakdown by customer tier for May 2025"""
        if 'customer_tier' not in self.df.columns:
            return {"error": "customer_tier column not found"}
        
        # Filter for May 2025
        may_2025_mask = (
            (self.df['order_date'].dt.month == 5) & 
            (self.df['order_date'].dt.year == 2025)
        )
        may_orders = self.df[may_2025_mask]
        
        # Count orders by tier
        tier_breakdown = may_orders['customer_tier'].value_counts().to_dict()
        
        # Revenue by tier
        revenue_by_tier = {}
        if 'order_value_usd' in self.df.columns:
            revenue_by_tier = may_orders.groupby('customer_tier')['order_value_usd'].sum().to_dict()
        
        # Average order value by tier
        avg_value_by_tier = {}
        if 'order_value_usd' in self.df.columns:
            avg_value_by_tier = may_orders.groupby('customer_tier')['order_value_usd'].mean().to_dict()
        
        return {
            "type": "customer_tier_analysis",
            "tier_breakdown": tier_breakdown,
            "revenue_by_tier": revenue_by_tier,
            "avg_value_by_tier": avg_value_by_tier,
            "total_orders": len(may_orders),
            "data": may_orders,
            "month": "May 2025"
        }
    
    def get_avg_order_value_by_category(self):
        """Get average order value by product category"""
        if 'product_category' not in self.df.columns or 'order_value_usd' not in self.df.columns:
            return {"error": "Required columns not found (product_category or order_value_usd)"}
        
        # Calculate average order value by category
        avg_by_category = self.df.groupby('product_category')['order_value_usd'].agg([
            'mean', 'count', 'sum'
        ]).round(2)
        
        avg_by_category.columns = ['avg_order_value', 'order_count', 'total_revenue']
        
        # Convert to dictionary
        result_dict = {}
        for category in avg_by_category.index:
            result_dict[category] = {
                'avg_order_value': avg_by_category.loc[category, 'avg_order_value'],
                'order_count': avg_by_category.loc[category, 'order_count'],
                'total_revenue': avg_by_category.loc[category, 'total_revenue']
            }
        
        return {
            "type": "avg_order_value_analysis",
            "avg_by_category": result_dict,
            "overall_avg": self.df['order_value_usd'].mean(),
            "data": avg_by_category
        }
    
    def get_general_overview(self):
        """Get general dataset overview"""
        total_orders = self.df['order_id'].nunique() if 'order_id' in self.df.columns else len(self.df)
        date_range = f"{self.df['order_date'].min().strftime('%Y-%m-%d')} to {self.df['order_date'].max().strftime('%Y-%m-%d')}" if 'order_date' in self.df.columns else "N/A"
        
        # Basic statistics
        stats = {
            "total_orders": total_orders,
            "date_range": date_range,
            "columns": list(self.df.columns),
            "data_types": self.df.dtypes.to_dict()
        }
        
        # Revenue summary
        if 'order_value_usd' in self.df.columns:
            stats["total_revenue"] = self.df['order_value_usd'].sum()
            stats["avg_order_value"] = self.df['order_value_usd'].mean()
        
        # Status summary
        if 'order_status' in self.df.columns:
            stats["status_breakdown"] = self.df['order_status'].value_counts().to_dict()
        
        # Customer type summary
        if 'customer_type' in self.df.columns:
            stats["customer_type_breakdown"] = self.df['customer_type'].value_counts().to_dict()
        
        # Delay summary
        if 'is_delayed' in self.df.columns:
            delayed_count = self.df['is_delayed'].sum()
            stats["delay_rate"] = (delayed_count / total_orders) * 100
            stats["total_delayed"] = delayed_count
        
        return {
            "type": "general_overview",
            "stats": stats,
            "data": self.df.head(100)  # Return sample data
        }