# 두 바운딩 박스의 겹치는 정도 표시
def calculate_overlap_ratio(box1,box2):
    
    
    x1_1,y1_1,x2_1,y2_1 = box1
    x1_2,y1_2,x2_2,y2_2 = box2
    
    # 겹치는 영역 계산
    overlap_x1 = max(x1_1,x1_2)
    overlap_y1 = max(y1_1,y1_2)
    overlap_x2 = min(x2_1,x2_2)
    overlap_y2 = min(y2_1,y2_2)
    
    # 겹치지 않는지 확인
    if overlap_x2 < overlap_x1 or overlap_y2 < overlap_y1:
        return 0.0
    
    # 영역 계산
    area_box1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_box2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
    
    # 겹치는 영역 비율 계산
    overlap_ratio = overlap_area / min(area_box1,area_box2)
    
    return overlap_ratio*100

# 겹치는 정도가 70 이상이면 하나 삭제
def filterd_and_del_bbox(bboxes):
    filtered_bbox = []
    n = len(bboxes)
    
    for i in range(n):
        current_bbox = bboxes[i]
        should_add_bbox = True
        
        for j in range(n):
            if i!=j:
                other_bbox = bboxes[j]
                overlap_ratio = calculate_overlap_ratio(current_bbox[:4],other_bbox[:4])
                
                if overlap_ratio >= 70:
                    if current_bbox[4] > other_bbox[4]:
                        should_add_bbox = False
                        break
                        
        if should_add_bbox:
            filtered_bbox.append(current_bbox)
            
    return filtered_bbox

# 흰색 픽셀 아닌 부분 비율 계
def cal_non_white_pixel_ratio(image):
    pixel_data = image.getdata()
    
    total_pixel = image.width * image.height
    non_white_pixel_count = 0
    
    for pixel in pixel_data:
        # 흰색이 아닌 경우 증가
        if pixel[:4] != (0,0,0,0):
            non_white_pixel_count += 1
            
    # 비율 계산
    non_white_pixel_ratio = non_white_pixel_count / total_pixel*100
    
    return non_white_pixel_ratio