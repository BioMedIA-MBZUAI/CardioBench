zero_shot_prompts = {
    "ejection_fraction": [
        "THE LEFT VENTRICULAR EJECTION FRACTION IS ESTIMATED TO BE <#>% ",
        "LV EJECTION FRACTION IS <#>%. ",
    ],
    "as_present": [
        "AORTIC STENOSIS IS PRESENT. ",
        "SEVERE AORTIC STENOSIS. ",
        "CALCIFIED AORTIC VALVE WITH RESTRICTED LEAFLET MOTION. ",
    ],
    "as_absent": [
        "NO AORTIC STENOSIS. ",
        "NO SIGNIFICANT AORTIC VALVE STENOSIS. ",
        "AORTIC VALVE OPENS NORMALLY WITHOUT STENOSIS. ",
    ],
    "asd_present": [
        "THERE IS AN ATRIAL SEPTAL DEFECT WITH LEFT TO RIGHT SHUNT. ",
        "ATRIAL SEPTAL DEFECT IS PRESENT. ",
        "ATRIAL SEPTAL DEFECT (ASD). ",
    ],
    "asd_absent": [
        "NO ATRIAL SEPTAL DEFECT IS SEEN. ",
        "NO EVIDENCE OF ATRIAL SEPTAL DEFECT. ",
        "INTACT ATRIAL SEPTUM WITHOUT DEFECT. ",
    ],
    "pah_present": [
        "FINDINGS ARE CONSISTENT WITH PULMONARY ARTERIAL HYPERTENSION. ",
        "ELEVATED PULMONARY ARTERY PRESSURE SUGGESTIVE OF PAH. ",
        "RIGHT VENTRICULAR PRESSURE OVERLOAD CONSISTENT WITH PAH. ",
    ],
    "pah_absent": [
        "NO EVIDENCE OF PULMONARY ARTERIAL HYPERTENSION. ",
        "PULMONARY ARTERY PRESSURE IS NOT ELEVATED. ",
        "NO SIGNS OF PAH. ",
    ],
    "wall_motion_abnormality": [
        "REGIONAL WALL MOTION IS ABNORMAL. ",
        "THERE IS EVIDENCE OF REGIONAL WALL MOTION ABNORMALITY. ",
        "REGIONAL WALL MOTION SHOWS HYPOKINESIS OR AKINESIS. ",
    ],
    "normal_wall_motion": [
        "REGIONAL WALL MOTION IS NORMAL. ",
        "NO EVIDENCE OF REGIONAL WALL MOTION ABNORMALITY. ",
    ],
    "stemi": [
        "FINDINGS ARE CONSISTENT WITH ST-ELEVATION MYOCARDIAL INFARCTION (STEMI). ",
        "ACUTE MYOCARDIAL INFARCTION WITH REGIONAL WALL MOTION ABNORMALITY. ",
        "ACUTE ST-ELEVATION MI. ",
    ],
    "no_stemi": [
        "NO EVIDENCE OF ST-ELEVATION MYOCARDIAL INFARCTION. ",
        "NO ACUTE MYOCARDIAL INFARCTION. ",
    ],
}



view_prompt = {
    "A2C": [
        "apical two-chamber echocardiography cine showing ONLY the left atrium and left ventricle (no right-sided chambers), mitral valve in profile",
        "apical 2-chamber ultrasound: left atrium + left ventricle, no right ventricle visible, long-axis of the LV",
        "A2C view echocardiogram with left atrium, left ventricle, anterior and inferior walls",
    ],
    "A3C": [
        "apical three-chamber (apical long-axis) echocardiography cine showing left ventricle, left atrium, and aortic valve with LVOT",
        "apical 3-chamber ultrasound: LV, LA, aortic root and valve in view; mitral and aortic valves seen together",
        "A3C view echocardiogram including LVOT and aortic valve (aka apical long-axis)",
    ],
    "A4C": [
        "apical four-chamber echocardiography cine showing all four chambers (LA, RA, LV, RV) with the interventricular septum vertical",
        "apical 4-chamber ultrasound: both atria and both ventricles visible, mitral and tricuspid valves seen",
        "A4C view echocardiogram with symmetric visualization of all four chambers",
    ],
    "PSAX": [
        "parasternal short-axis echocardiography cine with circular left ventricle and visible papillary muscles (cross-sectional view)",
        "PSAX ultrasound at mid-ventricle: round LV, papillary muscles, doughnut appearance; no four-chamber layout",
        "parasternal short-axis view showing concentric LV walls in cross-section",
    ],
    "PLAX": [
        "parasternal long-axis echocardiography cine showing left ventricle, left atrium, mitral valve, and aortic root in a longitudinal plane",
        "PLAX ultrasound: LV, LA, mitral valve, and LVOT aligned; long-axis, not circular cross-section",
        "parasternal long-axis view with mitral valve anterior leaflet and aortic valve continuity",
    ],
}
