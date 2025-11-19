"""
SGPA Calculator for VTU Results
Based on VTU grading system: https://vtustudent.com/vtu-sgpa-calculator/
"""


class SGPACalculator:
    # VTU Grade Point mapping
    GRADE_POINTS = {
        'S': 10, 'A': 9, 'B': 8, 'C': 7, 'D': 6, 'E': 5, 'F': 0, 'P': 5
    }
    
    # Marks to Grade conversion (VTU system)
    @staticmethod
    def marks_to_grade(marks):
        """Convert marks to grade based on VTU system"""
        if marks >= 90:
            return 'S', 10
        elif marks >= 80:
            return 'A', 9
        elif marks >= 70:
            return 'B', 8
        elif marks >= 60:
            return 'C', 7
        elif marks >= 50:
            return 'D', 6
        elif marks >= 40:
            return 'E', 5
        else:
            return 'F', 0
    
    @staticmethod
    def calculate_sgpa(subjects_data):
        """
        Calculate SGPA from subjects data
        
        Args:
            subjects_data: List of dicts with keys: 'total_marks', 'credits' (optional)
            
        Returns:
            dict: {
                'sgpa': float,
                'total_credits': int,
                'total_marks': int,
                'grade_points_earned': float
            }
        """
        total_grade_points = 0
        total_credits = 0
        total_marks = 0
        
        for subject in subjects_data:
            marks = subject.get('total_marks', 0)
            # If credits not provided, assume 4 credits per theory subject
            credits = subject.get('credits', 4)
            
            grade, grade_point = SGPACalculator.marks_to_grade(marks)
            
            total_grade_points += grade_point * credits
            total_credits += credits
            total_marks += marks
        
        sgpa = total_grade_points / total_credits if total_credits > 0 else 0
        
        return {
            'sgpa': round(sgpa, 2),
            'total_credits': total_credits,
            'total_marks': total_marks,
            'grade_points_earned': round(total_grade_points, 2)
        }
    
    @staticmethod
    def get_marks_color(marks):
        """Return color code based on marks (for UI display)"""
        if marks >= 90:
            return '#10b981'  # green-500 (Outstanding)
        elif marks >= 80:
            return '#3b82f6'  # blue-500 (Excellent)
        elif marks >= 70:
            return '#8b5cf6'  # purple-500 (Very Good)
        elif marks >= 60:
            return '#f59e0b'  # amber-500 (Good)
        elif marks >= 50:
            return '#f97316'  # orange-500 (Average)
        elif marks >= 40:
            return '#eab308'  # yellow-600 (Pass)
        else:
            return '#ef4444'  # red-500 (Fail)
    
    @staticmethod
    def get_performance_message(sgpa, failed_subjects):
        """Generate performance message based on results"""
        if len(failed_subjects) > 0:
            failed_names = ', '.join([s['name'] for s in failed_subjects])
            return f"⚠️ You have failed in {len(failed_subjects)} subject(s): {failed_names}"
        elif sgpa >= 9.0:
            return "🌟 Outstanding Performance! Keep up the excellent work!"
        elif sgpa >= 8.0:
            return "🎉 Excellent Performance! Well done!"
        elif sgpa >= 7.0:
            return "👍 Very Good Performance! Keep it up!"
        elif sgpa >= 6.0:
            return "✅ Good Performance! You passed all subjects!"
        else:
            return "✅ You have passed all subjects! Keep working hard!"
