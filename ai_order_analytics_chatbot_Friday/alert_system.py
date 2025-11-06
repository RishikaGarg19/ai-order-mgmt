
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AlertSystem:
    def __init__(self, df):
        self.df = df
        self.current_date = datetime.now()
        self.alerts = []
        
    def generate_alerts(self):
        """Generate all types of alerts based on current data"""
        self.alerts = []
        
        # Performance alerts
        self._check_delay_rate_alerts()
        self._check_revenue_decline_alerts()
        self._check_customer_satisfaction_alerts()
        
        # Operational alerts
        self._check_inventory_alerts()
        self._check_capacity_alerts()
        self._check_overdue_orders_alerts()
        
        # Financial alerts
        self._check_high_value_order_risks()
        self._check_cash_flow_alerts()
        
        # Quality alerts
        self._check_quality_issues()
        self._check_supplier_performance()
        
        # Sort alerts by severity and timestamp
        self.alerts.sort(key=lambda x: (
            {'critical': 0, 'warning': 1, 'info': 2}[x['severity']], 
            x['timestamp']
        ), reverse=True)
        
        return self.alerts
    
    def _add_alert(self, message, severity='info', category='general', metric_value=None, threshold=None):
        """Add an alert to the alerts list"""
        alert = {
            'message': message,
            'severity': severity,  # critical, warning, info
            'category': category,
            'timestamp': self.current_date,
            'metric_value': metric_value,
            'threshold': threshold
        }
        self.alerts.append(alert)
    
    def _check_delay_rate_alerts(self):
        """Check for delivery performance anomalies"""
        if 'is_delayed' not in self.df.columns:
            return
        
        # Overall delay rate
        total_orders = len(self.df)
        delayed_orders = self.df['is_delayed'].sum()
        delay_rate = (delayed_orders / total_orders) * 100 if total_orders > 0 else 0
        
        if delay_rate > 30:
            self._add_alert(
                f"CRITICAL: Delivery performance degraded - {delay_rate:.1f}% delay rate exceeds acceptable threshold",
                'critical', 'operations', delay_rate, 30
            )
        elif delay_rate > 20:
            self._add_alert(
                f"WARNING: Delivery performance concern - {delay_rate:.1f}% delay rate approaching critical levels",
                'warning', 'operations', delay_rate, 20
            )
        
        # Recent delay trend analysis
        if 'order_date' in self.df.columns:
            recent_date = self.current_date - timedelta(days=30)
            recent_orders = self.df[self.df['order_date'] >= recent_date]
            
            if len(recent_orders) > 0:
                recent_delay_rate = (recent_orders['is_delayed'].sum() / len(recent_orders)) * 100
                
                if recent_delay_rate > delay_rate + 10:
                    self._add_alert(
                        f"WARNING: Performance deterioration detected - Recent period: {recent_delay_rate:.1f}% vs baseline: {delay_rate:.1f}%",
                        'warning', 'operations'
                    )
    
    def _check_revenue_decline_alerts(self):
        """Check for revenue performance indicators"""
        if 'order_value_usd' not in self.df.columns or 'order_date' not in self.df.columns:
            return
        
        # Monthly revenue comparison
        current_month = self.current_date.month
        current_year = self.current_date.year
        
        # Get last month's data
        if current_month == 1:
            last_month = 12
            last_year = current_year - 1
        else:
            last_month = current_month - 1
            last_year = current_year
        
        current_month_revenue = self.df[
            (self.df['order_date'].dt.month == current_month) & 
            (self.df['order_date'].dt.year == current_year)
        ]['order_value_usd'].sum()
        
        last_month_revenue = self.df[
            (self.df['order_date'].dt.month == last_month) & 
            (self.df['order_date'].dt.year == last_year)
        ]['order_value_usd'].sum()
        
        # if last_month_revenue > 0:
        #     revenue_change = ((current_month_revenue - last_month_revenue) / last_month_revenue) * 100
            
        #     if revenue_change < -20:
        #         self._add_alert(
        #             f"CRITICAL: Significant revenue decline - {abs(revenue_change):.1f}% reduction from previous period",
        #             'critical', 'financial', revenue_change, -20
        #         )
        #     elif revenue_change < -10:
        #         self._add_alert(
        #             f"WARNING: Revenue performance below expectations - {abs(revenue_change):.1f}% decline from previous period",
        #             'warning', 'financial', revenue_change, -10
        #         )
    
    def _check_customer_satisfaction_alerts(self):
        """Check for customer relationship indicators"""
        if 'customer_satisfaction' not in self.df.columns:
            return
        
        # Overall satisfaction metrics
        avg_satisfaction = self.df['customer_satisfaction'].mean()
        
        if avg_satisfaction < 3.0:
            self._add_alert(
                f"CRITICAL: Customer satisfaction index critically low - {avg_satisfaction:.1f}/5.0 requires immediate intervention",
                'critical', 'customer', avg_satisfaction, 3.0
            )
        elif avg_satisfaction < 3.5:
            self._add_alert(
                f"WARNING: Customer satisfaction below acceptable standards - {avg_satisfaction:.1f}/5.0 performance threshold",
                'warning', 'customer', avg_satisfaction, 3.5
            )
        
        # Premium customer satisfaction analysis
        if 'customer_tier' in self.df.columns and 'order_value_usd' in self.df.columns:
            premium_customers = self.df[self.df['customer_tier'].isin(['Platinum', 'Gold'])]
            if len(premium_customers) > 0:
                premium_satisfaction = premium_customers['customer_satisfaction'].mean()
                
                if premium_satisfaction < 4.0:
                    self._add_alert(
                        f"WARNING: Premium customer satisfaction concern - {premium_satisfaction:.1f}/5.0 for high-value accounts",
                        'warning', 'customer'
                    )
    
    def _check_inventory_alerts(self):
        """Check for supply chain performance indicators"""
        if 'parts_availability' not in self.df.columns:
            return
        
        # Parts availability analysis
        low_availability_orders = self.df[self.df['parts_availability'] < 80]
        
        if len(low_availability_orders) > 0:
            affected_percentage = (len(low_availability_orders) / len(self.df)) * 100
            
            if affected_percentage > 20:
                self._add_alert(
                    f"CRITICAL: Supply chain constraint - {affected_percentage:.1f}% of orders affected by inventory shortages",
                    'critical', 'supply_chain', affected_percentage, 20
                )
            elif affected_percentage > 10:
                self._add_alert(
                    f"WARNING: Inventory management concern - {affected_percentage:.1f}% of orders experiencing supply constraints",
                    'warning', 'supply_chain', affected_percentage, 10
                )
        
        # Supply risk assessment
        if 'supply_risk_score' in self.df.columns:
            high_risk_orders = self.df[self.df['supply_risk_score'] > 7]
            
            if len(high_risk_orders) > 0:
                risk_percentage = (len(high_risk_orders) / len(self.df)) * 100
                
                if risk_percentage > 15:
                    self._add_alert(
                        f"WARNING: Elevated supply chain risk - {risk_percentage:.1f}% of orders in high-risk category",
                        'warning', 'supply_chain'
                    )
    
    def _check_capacity_alerts(self):
        """Check for production capacity indicators"""
        if 'capacity_available_percent' not in self.df.columns:
            return
        
        # Production capacity analysis
        low_capacity_orders = self.df[self.df['capacity_available_percent'] < 20]
        
        if len(low_capacity_orders) > 0:
            capacity_percentage = (len(low_capacity_orders) / len(self.df)) * 100
            
            if capacity_percentage > 25:
                self._add_alert(
                    f"CRITICAL: Production capacity constraint - {capacity_percentage:.1f}% of orders affected by resource limitations",
                    'critical', 'production', capacity_percentage, 25
                )
            elif capacity_percentage > 15:
                self._add_alert(
                    f"WARNING: Capacity utilization concern - {capacity_percentage:.1f}% of orders experiencing resource constraints",
                    'warning', 'production', capacity_percentage, 15
                )
        
        # Average capacity utilization analysis
        avg_capacity = self.df['capacity_available_percent'].mean()
        
        if avg_capacity < 25:
            self._add_alert(
                f"WARNING: Suboptimal resource allocation - Average capacity utilization at {avg_capacity:.1f}%",
                'warning', 'production'
            )
    
    def _check_overdue_orders_alerts(self):
        """Check for delivery schedule adherence"""
        if 'requested_delivery_date' not in self.df.columns:
            return
        
        # Schedule adherence analysis
        overdue_orders = self.df[self.df['requested_delivery_date'] < self.current_date]
        
        if len(overdue_orders) > 0:
            overdue_percentage = (len(overdue_orders) / len(self.df)) * 100
            
            # # High-value overdue analysis
            # if 'order_value_usd' in self.df.columns:
            #     high_value_overdue = overdue_orders[overdue_orders['order_value_usd'] > 500000]
                
            #     if len(high_value_overdue) > 0:
            #         total_value_at_risk = high_value_overdue['order_value_usd'].sum()
            #         self._add_alert(
            #             f"CRITICAL: High-value delivery failures - {len(high_value_overdue)} orders overdue, ${total_value_at_risk:,.0f} revenue exposure",
            #             'critical', 'operations'
            #         )
            
            # if overdue_percentage > 10:
            #     self._add_alert(
            #         f"WARNING: Schedule adherence below standards - {overdue_percentage:.1f}% of orders past delivery commitment",
            #         'warning', 'operations', overdue_percentage, 10
            #     )
    
    def _check_high_value_order_risks(self):
        """Check for strategic account risk indicators"""
        if 'order_value_usd' not in self.df.columns:
            return
        
        # Strategic account delivery risk
        high_value_orders = self.df[self.df['order_value_usd'] > 1000000]
        
        if len(high_value_orders) > 0 and 'is_delayed' in self.df.columns:
            delayed_high_value = high_value_orders[high_value_orders['is_delayed'] == True]
            
            if len(delayed_high_value) > 0:
                total_at_risk = delayed_high_value['order_value_usd'].sum()
                self._add_alert(
                    f"CRITICAL: High-value delivery delays - {len(delayed_high_value)} orders delayed, ${total_at_risk:,.0f} revenue at risk",
                    'critical', 'financial'
                )
        
        # Engineering approval bottlenecks for strategic accounts
        if 'engineering_status' in self.df.columns:
            pending_engineering = high_value_orders[high_value_orders['engineering_status'] == 'Pending']
            
            if len(pending_engineering) > 0:
                pending_value = pending_engineering['order_value_usd'].sum()
                self._add_alert(
                    f"WARNING: Engineering approval bottleneck - ${pending_value:,.0f} in strategic orders awaiting technical approval",
                    'warning', 'engineering'
                )
    
    def _check_cash_flow_alerts(self):
        """Check for financial performance indicators"""
        if 'payment_terms_days' not in self.df.columns or 'order_value_usd' not in self.df.columns:
            return
        
        # Payment terms exposure analysis
        extended_terms = self.df[self.df['payment_terms_days'] > 60]
        
        if len(extended_terms) > 0:
            extended_value = extended_terms['order_value_usd'].sum()
            total_value = self.df['order_value_usd'].sum()
            
            if total_value > 0:
                extended_percentage = (extended_value / total_value) * 100
                
                if extended_percentage > 30:
                    self._add_alert(
                        f"WARNING: Cash flow exposure - {extended_percentage:.1f}% of order value on extended payment terms",
                        'warning', 'financial'
                    )
    
    def _check_quality_issues(self):
        """Check for quality assurance indicators"""
        if 'quality_score' not in self.df.columns:
            return
        
        # Quality performance analysis
        low_quality_orders = self.df[self.df['quality_score'] < 85]
        
        if len(low_quality_orders) > 0:
            quality_percentage = (len(low_quality_orders) / len(self.df)) * 100
            
            if quality_percentage > 15:
                self._add_alert(
                    f"WARNING: Quality assurance concern - {quality_percentage:.1f}% of orders below quality standards",
                    'warning', 'quality', quality_percentage, 15
                )
        
        # Rework impact analysis
        if 'rework_required' in self.df.columns:
            rework_orders = self.df[self.df['rework_required'] == True]
            
            if len(rework_orders) > 0:
                rework_percentage = (len(rework_orders) / len(self.df)) * 100
                
                if rework_percentage > 10:
                    self._add_alert(
                        f"WARNING: Process efficiency concern - {rework_percentage:.1f}% of orders requiring rework intervention",
                        'warning', 'quality'
                    )
    
    def _check_supplier_performance(self):
        """Check for vendor performance indicators"""
        if 'supply_risk_score' not in self.df.columns:
            return
        
        # Critical supplier risk concentration
        critical_risk_orders = self.df[self.df['supply_risk_score'] > 8]
        
        if len(critical_risk_orders) > 0:
            risk_percentage = (len(critical_risk_orders) / len(self.df)) * 100
            
            if risk_percentage > 20:
                self._add_alert(
                    f"CRITICAL: Vendor performance crisis - {risk_percentage:.1f}% of orders experiencing critical supplier risk",
                    'critical', 'supply_chain'
                )
    
    def get_alerts_by_category(self, category):
        """Get alerts filtered by category"""
        return [alert for alert in self.alerts if alert['category'] == category]
    
    def get_alerts_by_severity(self, severity):
        """Get alerts filtered by severity"""
        return [alert for alert in self.alerts if alert['severity'] == severity]
    
    def get_alert_summary(self):
        """Get a summary of all alerts"""
        critical_count = len(self.get_alerts_by_severity('critical'))
        warning_count = len(self.get_alerts_by_severity('warning'))
        info_count = len(self.get_alerts_by_severity('info'))
        
        return {
            'total_alerts': len(self.alerts),
            'critical': critical_count,
            'warning': warning_count,
            'info': info_count
        }