import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from streamlit_cropper import st_cropper

class NailHealthAnalyzer:
    """
    Nail Health Analyzer - Background Removal + Easy Crop + Full Visual Analysis
    """
    
    def __init__(self):
        self.image = None
        self.original_image = None
        self.background_removed = None
        self.cropped_image = None
        self.nail_mask = None
        self.isolated_nail = None
        self.hemoglobin_results = {}
        self.oxygen_results = {}
        self.iron_results = {}
        self.overall_status = ""
        
    def load_image(self, image):
        """Load and remove background"""
        try:
            self.original_image = np.array(image)
            if self.original_image is None:
                raise ValueError("Failed to load image")
            
            # Convert RGB to BGR for OpenCV if needed
            if len(self.original_image.shape) == 3:
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR)
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            
            st.success("✓ Image loaded successfully!")
            return True
        except Exception as e:
            st.error(f"✗ Error loading image: {e}")
            return False
    
    def remove_background(self):
        """
        إزالة الخلفية تلقائياً - عزل الإصبع والظفر فقط
        """
        img = self.original_image.copy()
        
        # 1. تحويل لـ HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # 2. تحديد نطاق لون البشرة/الظفر
        lower_skin = np.array([0, 10, 60])
        upper_skin = np.array([25, 255, 255])
        
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # 3. استخدام Otsu's thresholding
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 4. دمج الماسكات
        combined_mask = cv2.bitwise_and(skin_mask, otsu_mask)
        
        # 5. تنظيف الماسك
        kernel = np.ones((7, 7), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # 6. إيجاد أكبر منطقة (الإصبع)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # إنشاء ماسك نهائي
            final_mask = np.zeros_like(gray)
            cv2.drawContours(final_mask, [largest_contour], -1, 255, -1)
            
            # تنعيم الحواف
            final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)
            
            # عزل الإصبع (إزالة الخلفية)
            self.background_removed = cv2.bitwise_and(img, img, mask=final_mask)
            
            #جعل الخلفية بيضاء
            white_bg = np.ones_like(img) * 255
            white_bg = cv2.bitwise_and(white_bg, white_bg, mask=cv2.bitwise_not(final_mask))
            self.background_removed = cv2.add(self.background_removed, white_bg)
            
            finger_percentage = (np.sum(final_mask > 0) / final_mask.size) * 100
            st.success(f"✓ Background removed! (Finger area: {finger_percentage:.1f}%)")
            
            return True
        else:
            st.warning("⚠️ Could not remove background, using original")
            self.background_removed = img.copy()
            return False
    
    def set_cropped_region(self, cropped_img):
        """
        تعيين المنطقة المقصوصة (الظفر المحدد)
        """
        self.cropped_image = np.array(cropped_img)
        
        # إنشاء ماسك كامل للمنطقة المقصوصة
        height, width = self.cropped_image.shape[:2]
        self.nail_mask = np.ones((height, width), dtype=np.uint8) * 255
        
        self.isolated_nail = self.cropped_image.copy()
        self.image = self.cropped_image.copy()
        
        st.success(f"✓ Nail selected! Size: {width}x{height} pixels")
        return True
    
    def analyze_nail_color_for_hemoglobin(self):
        """Analyze nail color for hemoglobin"""
        nail_pixels = self.image.reshape(-1, 3)
        hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        hsv_pixels = hsv.reshape(-1, 3)
        
        avg_r = np.mean(nail_pixels[:, 0])
        avg_g = np.mean(nail_pixels[:, 1])
        avg_b = np.mean(nail_pixels[:, 2])
        
        avg_saturation = np.mean(hsv_pixels[:, 1]) / 255 * 100
        pinkness = avg_r - ((avg_g + avg_b) / 2)
        brightness = np.mean(nail_pixels) / 255 * 100
        
        hemoglobin_score = 0
        
        if pinkness > 20 and avg_saturation > 25 and brightness < 65:
            status = "Normal 🟢"
            description = "Healthy pink color - Normal hemoglobin"
            hemoglobin_score = 85 + (pinkness / 5)
        elif pinkness > 10 and avg_saturation > 15:
            status = "Slightly Low 🟡"
            description = "Slightly pale - Possible mild deficiency"
            hemoglobin_score = 55 + (pinkness * 1.5)
        else:
            status = "Low 🔴"
            description = "Pale/whitish - Hemoglobin deficiency"
            hemoglobin_score = 25 + (pinkness * 2)
        
        hemoglobin_score = max(15, min(100, hemoglobin_score))
        estimated_hb = 12 + ((hemoglobin_score - 50) / 10)
        estimated_hb = max(7, min(18, estimated_hb))
        
        return {
            'status': status,
            'score': round(hemoglobin_score, 2),
            'estimated_hb': round(estimated_hb, 1),
            'description': description,
            'pinkness': round(pinkness, 2),
            'saturation': round(avg_saturation, 2),
            'brightness': round(brightness, 2),
            'avg_colors': {
                'R': round(avg_r, 2),
                'G': round(avg_g, 2),
                'B': round(avg_b, 2)
            }
        }
    
    def calculate_oxygen_saturation(self):
        """Calculate oxygen saturation"""
        avg_r = self.hemoglobin_results['avg_colors']['R']
        avg_g = self.hemoglobin_results['avg_colors']['G']
        avg_b = self.hemoglobin_results['avg_colors']['B']
        
        pinkness = self.hemoglobin_results['pinkness']
        saturation = self.hemoglobin_results['saturation']
        brightness = self.hemoglobin_results['brightness']
        
        rb_ratio = avg_r / (avg_b + 1)
        cyanosis_index = avg_b - avg_r
        
        spo2 = 100.0
        
        if rb_ratio < 0.9:
            spo2 -= 25
        elif rb_ratio < 1.1:
            spo2 -= 15
        elif rb_ratio < 1.3:
            spo2 -= 8
        elif rb_ratio < 1.5:
            spo2 -= 3
        
        if cyanosis_index > 10:
            spo2 -= 20
        elif cyanosis_index > 5:
            spo2 -= 10
        elif cyanosis_index > 0:
            spo2 -= 5
        
        if brightness > 75:
            spo2 -= 8
        elif brightness > 68:
            spo2 -= 4
        
        if saturation < 15:
            spo2 -= 10
        elif saturation < 20:
            spo2 -= 5
        
        if pinkness < 5:
            spo2 -= 12
        elif pinkness < 10:
            spo2 -= 6
        elif pinkness < 15:
            spo2 -= 3
        
        spo2 = max(70, min(100, spo2))
        
        if spo2 >= 95:
            status = "Normal 🟢"
            description = "Healthy oxygen saturation"
        elif spo2 >= 90:
            status = "Borderline 🟡"
            description = "Mild desaturation possible"
        elif spo2 >= 85:
            status = "Low 🟠"
            description = "Moderate hypoxemia"
        else:
            status = "Critical 🔴"
            description = "Severe hypoxemia"
        
        return {
            'spo2': round(spo2, 1),
            'status': status,
            'description': description,
            'rb_ratio': round(rb_ratio, 2),
            'cyanosis_index': round(cyanosis_index, 2),
            'has_cyanosis': cyanosis_index > 5
        }
    
    def detect_lunula(self):
        """Detect white lunula"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        _, white_mask = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
        
        height = self.image.shape[0]
        lower_third = white_mask[int(height * 0.65):, :]
        
        white_pixels = np.sum(lower_third == 255)
        total_pixels = lower_third.size
        lunula_percentage = (white_pixels / total_pixels) * 100
        
        if lunula_percentage > 12:
            status = "Clear and Normal 🟢"
            score = 95
        elif lunula_percentage > 5:
            status = "Small 🟡"
            score = 65
        else:
            status = "Absent/Very Small 🔴"
            score = 35
        
        return {
            'status': status,
            'score': score,
            'percentage': round(lunula_percentage, 2)
        }
    
    def detect_paleness(self):
        """Measure paleness"""
        brightness = np.mean(self.image)
        brightness_percentage = (brightness / 255) * 100
        
        if brightness_percentage < 55:
            status = "Normal Color 🟢"
            score = 95
        elif brightness_percentage < 68:
            status = "Moderate Paleness 🟡"
            score = 65
        else:
            status = "Very Pale 🔴"
            score = 35
        
        return {
            'status': status,
            'score': score,
            'brightness': round(brightness_percentage, 2)
        }
    
    def detect_white_spots(self):
        """Detect white spots"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        spots_count = sum(1 for c in contours if 5 < cv2.contourArea(c) < 400)
        
        if spots_count == 0:
            status = "No Spots 🟢"
            score = 95
        elif spots_count <= 2:
            status = "Few Spots 🟡"
            score = 75
        elif spots_count <= 5:
            status = "Moderate 🟡"
            score = 55
        else:
            status = "Many Spots 🔴"
            score = 35
        
        return {
            'status': status,
            'score': score,
            'count': spots_count
        }
    
    def detect_texture_and_shape(self):
        """Detect texture"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
        
        edge_density = np.sum(edges > 0) / edges.size * 100
        texture_roughness = (np.mean(np.abs(sobelx)) + np.mean(np.abs(sobely))) / 2
        
        vertical_edges = np.abs(sobelx)
        vertical_intensity = np.mean(vertical_edges[vertical_edges > 20])
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        concavity_score = 0
        
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 100:
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                contour_area = cv2.contourArea(largest_contour)
                if hull_area > 0:
                    concavity_score = ((hull_area - contour_area) / hull_area) * 100
        
        texture_score = (
            edge_density * 1.2 +
            (texture_roughness / 8) +
            concavity_score * 1.5 +
            (vertical_intensity / 30 if not np.isnan(vertical_intensity) else 0)
        )
        
        if texture_score < 12:
            status = "Smooth and Normal 🟢"
            score = 95
        elif texture_score < 22:
            status = "Mild Ridges 🟡"
            score = 70
        elif texture_score < 35:
            status = "Moderate Ridges 🟡"
            score = 50
        else:
            status = "Rough/Concave 🔴"
            score = 30
        
        return {
            'status': status,
            'score': score,
            'edge_density': round(edge_density, 2),
            'texture_roughness': round(texture_roughness, 2),
            'concavity': round(concavity_score, 2),
            'vertical_intensity': round(vertical_intensity, 2) if not np.isnan(vertical_intensity) else 0,
            'texture_score': round(texture_score, 2)
        }
    
    def determine_overall_status(self):
        """Determine overall status"""
        hemoglobin_status = self.hemoglobin_results['status']
        
        critical_issues = 0
        moderate_issues = 0
        
        if "Low 🔴" in hemoglobin_status:
            critical_issues += 2
        elif "Slightly Low 🟡" in hemoglobin_status:
            moderate_issues += 1
        
        for param in ['lunula', 'paleness', 'white_spots', 'texture']:
            score = self.iron_results[param]['score']
            if score < 50:
                critical_issues += 1
            elif score < 70:
                moderate_issues += 1
        
        if critical_issues >= 2:
            self.overall_status = "🔴 ABNORMAL - Requires medical attention"
            recommendation = "Please consult a doctor for blood tests"
        elif critical_issues >= 1 or moderate_issues >= 3:
            self.overall_status = "🟡 BORDERLINE - Recommended to consult doctor"
            recommendation = "Consider checking with healthcare provider"
        else:
            self.overall_status = "🟢 NORMAL - Healthy nail indicators"
            recommendation = "Maintain healthy diet and monitor regularly"
        
        return self.overall_status, recommendation
    
    def analyze(self):
        """Comprehensive analysis"""
        if self.image is None:
            st.error("Please crop nail first!")
            return None
        
        with st.spinner("🧬 Analyzing..."):
            self.hemoglobin_results = self.analyze_nail_color_for_hemoglobin()
            self.oxygen_results = self.calculate_oxygen_saturation()
            
            lunula = self.detect_lunula()
            paleness = self.detect_paleness()
            white_spots = self.detect_white_spots()
            texture = self.detect_texture_and_shape()
            
            iron_score = (
                lunula['score'] * 0.25 +
                paleness['score'] * 0.30 +
                white_spots['score'] * 0.15 +
                texture['score'] * 0.30
            )
            
            self.iron_results = {
                'iron_score': round(iron_score, 2),
                'lunula': lunula,
                'paleness': paleness,
                'white_spots': white_spots,
                'texture': texture,
                'status': self.get_iron_status(iron_score)
            }
        
        overall_status, recommendation = self.determine_overall_status()
        
        return {
            'hemoglobin': self.hemoglobin_results,
            'oxygen': self.oxygen_results,
            'iron': self.iron_results,
            'overall_status': overall_status,
            'recommendation': recommendation
        }
    
    def get_iron_status(self, score):
        if score >= 85:
            return "Normal Level 🟢"
        elif score >= 70:
            return "Borderline 🟡"
        elif score >= 55:
            return "Possible Mild Deficiency 🟡"
        else:
            return "Possible Iron Deficiency 🔴"

def main():
    st.set_page_config(
        page_title="Nail Health Analyzer",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 Nail Health Analyzer")
    st.markdown("### Auto Background Removal + Easy Crop + Full Visual Analysis")
    
    # Sidebar
    st.sidebar.title("📋 How to Use")
    st.sidebar.info("""
    **3 Easy Steps:**
    
    1. 📤 Upload nail image
       → Background removed automatically
    
    2. ✂️ Crop the nail area
       → Drag box, then click Crop
    
    3. 🔬 Click Analyze
       → Get full visual analysis
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.warning("""
    ⚠️ **Medical Disclaimer**
    
    Educational purposes only.
    Consult a doctor for diagnosis.
    Get proper blood tests (CBC).
    """)
    
    # Initialize
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'bg_removed' not in st.session_state:
        st.session_state.bg_removed = False
    if 'cropped' not in st.session_state:
        st.session_state.cropped = False
    
    # Upload
    uploaded_file = st.file_uploader(
        "📁 Upload Nail Image", 
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        if st.session_state.analyzer is None:
            st.session_state.analyzer = NailHealthAnalyzer()
            st.session_state.analyzer.load_image(image)
            
            # Remove background automatically
            with st.spinner("🔄 Removing background..."):
                st.session_state.analyzer.remove_background()
                st.session_state.bg_removed = True
        
        analyzer = st.session_state.analyzer
        
        # Show before/after background removal
        if st.session_state.bg_removed and not st.session_state.cropped:
            st.markdown("---")
            st.subheader("✅ Step 1: Background Removed")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(analyzer.original_image, caption="Original", use_container_width=True)
            
            with col2:
                st.image(analyzer.background_removed, caption="Background Removed", use_container_width=True)
            
            st.markdown("---")
            st.subheader("✂️ Step 2: Crop the Nail")
            st.info("👆 **Drag the blue box around the nail only**")
            
            # Cropper
            bg_removed_pil = Image.fromarray(analyzer.background_removed)
            cropped_img = st_cropper(
                bg_removed_pil,
                realtime_update=True,
                box_color='#0066FF',
                aspect_ratio=None
            )
            
            col1, col2 = st.columns([1, 4])
            
            with col1:
                if st.button("✅ Crop Image", type="primary", use_container_width=True):
                    analyzer.set_cropped_region(cropped_img)
                    st.session_state.cropped = True
                    st.rerun()
        
        elif st.session_state.cropped:
            # Show cropped result
            st.markdown("---")
            st.subheader("✅ Cropped Nail Ready")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(analyzer.original_image, caption="1. Original", use_container_width=True)
            
            with col2:
                st.image(analyzer.background_removed, caption="2. BG Removed", use_container_width=True)
            
            with col3:
                st.image(analyzer.cropped_image, caption="3. Cropped Nail", use_container_width=True)
            
            col_recrop, col_analyze = st.columns([1, 2])
            
            with col_recrop:
                if st.button("🔄 Re-crop", use_container_width=True):
                    st.session_state.cropped = False
                    st.rerun()
            
            with col_analyze:
                if st.button("🔬 Analyze Nail", type="primary", use_container_width=True):
                    results = analyzer.analyze()
                    
                    if results:
                        st.success("✅ Analysis completed!")
                        
                        # Results
                        st.markdown("---")
                        st.header("📊 Analysis Results")
                        
                        st.markdown(f"### {results['overall_status']}")
                        st.info(f"💡 {results['recommendation']}")
                        
                        # Tabs
                        tab1, tab2, tab3, tab4 = st.tabs([
                            "🧬 Hemoglobin", 
                            "💨 Oxygen", 
                            "⚙️ Iron",
                            "📈 Visual Analysis"
                        ])
                        
                        with tab1:
                            hb = results['hemoglobin']
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Hemoglobin Score", f"{hb['score']}%")
                                st.metric("Estimated Hb", f"{hb['estimated_hb']} g/dL")
                                st.metric("Status", hb['status'])
                            
                            with col2:
                                st.subheader("Color Metrics")
                                st.write(f"**Pinkness:** {hb['pinkness']} (Normal > 20)")
                                st.write(f"**Saturation:** {hb['saturation']}% (Normal > 25%)")
                                st.write(f"**Brightness:** {hb['brightness']}% (Normal < 65%)")
                                st.write(f"**Description:** {hb['description']}")
                        
                        with tab2:
                            ox = results['oxygen']
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("SpO2", f"{ox['spo2']}%")
                                st.metric("Status", ox['status'])
                            
                            with col2:
                                st.subheader("Technical Metrics")
                                st.write(f"**R/B Ratio:** {ox['rb_ratio']} (Normal > 1.5)")
                                st.write(f"**Cyanosis Index:** {ox['cyanosis_index']} (Normal < 0)")
                                st.write(f"**Cyanosis:** {'Yes ⚠️' if ox['has_cyanosis'] else 'No ✓'}")
                                st.write(f"**Description:** {ox['description']}")
                        
                        with tab3:
                            iron = results['iron']
                            st.metric("Iron Score", f"{iron['iron_score']}%")
                            st.metric("Status", iron['status'])
                            
                            cols = st.columns(4)
                            params = [
                                ("Lunula", iron['lunula']),
                                ("Paleness", iron['paleness']),
                                ("Spots", iron['white_spots']),
                                ("Texture", iron['texture'])
                            ]
                            
                            for col, (name, data) in zip(cols, params):
                                with col:
                                    st.metric(name, f"{data['score']}%")
                                    st.caption(data['status'])
                        
                        with tab4:
                            st.subheader("📈 Complete Visual Analysis")
                            
                            # Create visualization
                            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
                            
                            # Row 1: Images
                            axes[0, 0].imshow(analyzer.original_image)
                            axes[0, 0].set_title('📷 Original')
                            axes[0, 0].axis('off')
                            
                            axes[0, 1].imshow(analyzer.background_removed)
                            axes[0, 1].set_title('🗑️ BG Removed')
                            axes[0, 1].axis('off')
                            
                            axes[0, 2].imshow(analyzer.cropped_image)
                            axes[0, 2].set_title('✂️ Cropped Nail')
                            axes[0, 2].axis('off')
                            
                            # Row 2: Heat maps
                            hsv = cv2.cvtColor(analyzer.cropped_image, cv2.COLOR_RGB2HSV)
                            
                            im1 = axes[1, 0].imshow(hsv[:, :, 1], cmap='hot')
                            axes[1, 0].set_title('🌡️ Saturation Map')
                            axes[1, 0].axis('off')
                            plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
                            
                            brightness = cv2.cvtColor(analyzer.cropped_image, cv2.COLOR_RGB2GRAY)
                            im2 = axes[1, 1].imshow(brightness, cmap='viridis')
                            axes[1, 1].set_title('💡 Brightness Map')
                            axes[1, 1].axis('off')
                            plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
                            
                            pink_map = analyzer.cropped_image[:,:,0] - (analyzer.cropped_image[:,:,1] + analyzer.cropped_image[:,:,2])/2
                            im3 = axes[1, 2].imshow(pink_map, cmap='RdPu')
                            axes[1, 2].set_title('🎨 Pinkness Map')
                            axes[1, 2].axis('off')
                            plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)
                            
                            # Row 3: Charts
                            hb = results['hemoglobin']
                            rgb_values = [hb['avg_colors']['R'], hb['avg_colors']['G'], hb['avg_colors']['B']]
                            colors = ['#ff6b6b', '#51cf66', '#339af0']
                            bars = axes[2, 0].bar(['Red', 'Green', 'Blue'], rgb_values, color=colors, alpha=0.8)
                            axes[2, 0].set_title('🎨 RGB Distribution')
                            axes[2, 0].set_ylabel('Intensity')
                            
                            for bar, value in zip(bars, rgb_values):
                                axes[2, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                                              f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
                            
                            # Iron parameters
                            iron_params = ['Lunula', 'Paleness', 'Spots', 'Texture']
                            iron_scores = [
                                results['iron']['lunula']['score'],
                                results['iron']['paleness']['score'],
                                results['iron']['white_spots']['score'],
                                results['iron']['texture']['score']
                            ]
                            iron_colors = ['#22c55e' if s > 80 else '#eab308' if s > 60 else '#ef4444' for s in iron_scores]
                            axes[2, 1].barh(iron_params, iron_scores, color=iron_colors)
                            axes[2, 1].set_title('⚙️ Iron Parameters')
                            axes[2, 1].set_xlim(0, 100)
                            axes[2, 1].set_xlabel('Score (%)')
                            
                            # Overall scores
                            scores_names = ['Hemoglobin', 'Oxygen', 'Iron']
                            scores_values = [hb['score'], results['oxygen']['spo2'], results['iron']['iron_score']]
                            score_colors = ['#3b82f6', '#10b981', '#f59e0b']
                            bars2 = axes[2, 2].bar(scores_names, scores_values, color=score_colors)
                            axes[2, 2].set_title('📊 Overall Scores')
                            axes[2, 2].set_ylim(0, 100)
                            axes[2, 2].set_ylabel('Score (%)')
                            
                            for bar, value in zip(bars2, scores_values):
                                axes[2, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                                              f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Additional detailed metrics
                            st.markdown("---")
                            st.subheader("🔬 Detailed Metrics")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("**🧬 Hemoglobin Details**")
                                st.write(f"• Red: {hb['avg_colors']['R']:.1f}")
                                st.write(f"• Green: {hb['avg_colors']['G']:.1f}")
                                st.write(f"• Blue: {hb['avg_colors']['B']:.1f}")
                                st.write(f"• Pinkness: {hb['pinkness']:.2f}")
                                st.write(f"• Saturation: {hb['saturation']:.1f}%")
                                st.write(f"• Brightness: {hb['brightness']:.1f}%")
                            
                            with col2:
                                st.markdown("**💨 Oxygen Details**")
                                ox = results['oxygen']
                                st.write(f"• SpO2: {ox['spo2']:.1f}%")
                                st.write(f"• R/B Ratio: {ox['rb_ratio']:.2f}")
                                st.write(f"• Cyanosis Index: {ox['cyanosis_index']:.2f}")
                                st.write(f"• Has Cyanosis: {'Yes' if ox['has_cyanosis'] else 'No'}")
                            
                            with col3:
                                st.markdown("**⚙️ Iron Details**")
                                iron = results['iron']
                                st.write(f"• Lunula: {iron['lunula']['percentage']:.1f}%")
                                st.write(f"• Paleness: {iron['paleness']['brightness']:.1f}%")
                                st.write(f"• White Spots: {iron['white_spots']['count']}")
                                st.write(f"• Texture Score: {iron['texture']['texture_score']:.2f}")
                                st.write(f"• Edge Density: {iron['texture']['edge_density']:.2f}%")
    
    else:
        st.info("👆 **Upload an image to start**")
        
        st.markdown("---")
        st.markdown("### 📸 Tips for Best Results:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**✅ Good Photo:**")
            st.write("• Natural daylight")
            st.write("• Clean, dry nail")
            st.write("• Straight angle")
            st.write("• Clear focus")
            st.write("• No nail polish")
        
        with col2:
            st.markdown("**🎯 What We Analyze:**")
            st.write("• Hemoglobin levels")
            st.write("• Oxygen saturation")
            st.write("• Iron indicators")
            st.write("• Color & texture")
            st.write("• Shape & ridges")
        
        with col3:
            st.markdown("**⚡ Features:**")
            st.write("• Auto background removal")
            st.write("• Easy crop tool")
            st.write("• Visual heat maps")
            st.write("• Detailed charts")
            st.write("• Full metrics")

if __name__ == "__main__":
    main()
