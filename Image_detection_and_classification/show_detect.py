import sys
import pixel
import datetime
import os
from collections import Counter
from IPython.display import display
from PIL import Image,ImageDraw,ImageFont
import warnings
warnings.filterwarnings(action='ignore')

# 2단계 추론을 위한 crop 이미지 저장용 디렉토리 생성
def create_drawing_crop_image_directory(queried_index, input_path_lastname):
    
    # 현재 날짜
    today_date = datetime.date.today().strftime('%Y-%m-%d')
    
    # 이전에 실행한 날짜와 현재 날짜 비교하여 숫자 초기화 여부 판단
    if not os.path.exists('/data/LG_ES_POC/002. PoC_개발/09. 핵심이미지 식별/another_workspace/04. 추론 및 후처리/drawing_inference_result/last_date.txt'):
        last_date = today_date
        count = 0
    else:
        with open('/data/LG_ES_POC/002. PoC_개발/09. 핵심이미지 식별/another_workspace/04. 추론 및 후처리/drawing_inference_result/last_date.txt','r') as file:
            last_date, count = file.read().split(',')
            
            if last_date != today_date:
                count = 0
                last_date = today_date
            else:
                count = int(count)
                
    # 디렉토리 생성
    directory_name = f'/data/LG_ES_POC/002. PoC_개발/09. 핵심이미지 식별/another_workspace/04. 추론 및 후처리/drawing_inference_result/{today_date}/{queried_index}/{input_path_lastname}'
    os.makedirs(directory_name, exist_ok=True)
    
    
    # 카운트 증가 및 파일 저장            
    count += 1
    with open('/data/LG_ES_POC/002. PoC_개발/09. 핵심이미지 식별/another_workspace/04. 추론 및 후처리/drawing_inference_result/last_date.txt','w') as file:
        file.write(f'{last_date},{count}')
        
    return directory_name

# 1차 분류용
def convert_cls(num):
    if num == 0:
        return 'Drawing'
    if num == 1:
        return "Table"
    if num == 2:
        return 'Col'

# 2차 분류용
def convert_cls_for_classification(num):
    if num == 0:
        return 'Ass’y'
    if num == 1:
        return "Cell"
    if num == 2:
        return 'Etc'
    if num == 3:
        return 'Electrode'
    if num == 4:
        return 'Roll'
    

# 2단계 추론을 위한 1단계 추론 함수 - Detection
def show_image_bbox_first_step(image_path, model, queried_index=None):
    # 메시지 숨기기
    warnings.filterwarnings(action='ignore')
    
    result = []
    crop_path = []
    confidence = []
    
    # yolo에 이미지 입력
    input_image = Image.open(image_path).convert('RGB')
    test_image_result = model(input_image)
    
    # 글자 크기 설정을 위한 설정
    h = input_image.height
    w = input_image.width
    
    # 폰트 설정
    font = ImageFont.truetype("/usr/share/fonts/truetype/lg-smart-fonts/LG Smart Bold.ttf", 30, encoding="UTF-8")
    
    # 이미지 그리기 설정
    draw = ImageDraw.Draw(input_image)
    # 겹치는 결과 확인 후 하나로 정리
    # filter_boxes = pixel.filterd_and_del_bbox(test_image_result[0].boxes.data)
    
    category_count = [0,0,0]
    
    for n,i in enumerate(test_image_result[0].boxes.data):
        if float(i[4]) >= 0.8:
            outline = (255,0,0)
            width = 3
        else:
            outline = (0,0,255)
            width = 3
            
        if float(i[4]>0.35):

            # 카테고리 카운트 증가    
            category_count[int(float(i[5]))] += 1
            
            # 추론 결과 저장
            result.append(convert_cls(int(float(i[5]))))
            # confidence 저장
            confidence.append(int(float(i[4]*100)))
            
            # 인식한 도면 이미지 자르기
            cropped_image = input_image.crop((float(i[0]),float(i[1]),float(i[2]),float(i[3])))
            cropped_image = cropped_image.resize((448, 448))
            # 자른 이미지 저장
            if int(i[5]) == 0:
                # 디렉토리 생성
                save_directory_drawing = create_drawing_crop_image_directory(queried_index, image_path.split('/')[-1])
                cropped_image.save(save_directory_drawing+f'/{queried_index}_' + str(n) + f'_{convert_cls(int(i[5]))}' + '.png')    
                crop_path.append(save_directory_drawing+f'/{queried_index}_' + str(n) + f'_{convert_cls(int(i[5]))}' + '.png')
                # 원본 이미지도 같이 저장
                #input_image.save(save_directory+f"/{image_path.split('/')[-1]}" + ".PNG")
            else:
                 pass

            # 바운딩 박스 그리기
            draw.rectangle((float(i[0]),float(i[1]),float(i[2]),float(i[3])),outline=outline,width =width)
            text = f'{convert_cls(int(float(i[5])))} - {int(float(i[4])*100)}%'
                
            draw.text(((float(i[0])+(float(i[0])*0.03)), float(i[1]) ), text, fill="black", font=font)

        else:
            pass
    
    if sum(category_count[:1]) > 0:
        text_category = f'Category : {convert_cls(category_count.index(max(category_count[:1])))}'
        draw.text((float(w)*0.3, float(h)*0.35 ), text_category, fill="black", font=font)
    else:
        text_category = f'Category : {convert_cls(category_count.index(max(category_count[1:])))}'
        draw.text((float(w)*0.3, float(h)*0.35 ), text_category, fill="black", font=font)
     
    
    crop_image = {
        'crop_path':crop_path,
        'confidence':confidence
    }
    # 추론 결과 저장
    result_count = Counter(result)
    # 추론 결과 정리
    inference_result = "\\n ".join([f"{key} {value}개" for key, value in result_count.items()])
    # 추론 결과 출력
    print(f"[추론 결과]\\n {inference_result}")

    display(input_image.resize((1280,1280)))
    return result_count, crop_image, input_image


# 2단계 추론 - Classify
def show_cropped_classified_image(image_path, model):
    # 메시지 숨기기
    warnings.filterwarnings(action='ignore')
    
    # 분류할 crop된 이미지
    input_image = Image.open(image_path).convert('RGB')
    
    crop_image_list = []
    classify_list = []
    confidence_list = []
    
    # 분류 모델 가동
    result = model(input_image)
    
    crop_image_list.append(image_path)
    classify_list.append(convert_cls_for_classification(result[0].probs.top1))
    confidence_list.append(result[0].probs.top1conf)
    
    # 폰트 설정
    font = ImageFont.truetype("/usr/share/fonts/truetype/lg-smart-fonts/LG Smart Bold.ttf", 30, encoding="UTF-8")
    # 이미지 그리기 설정
    draw = ImageDraw.Draw(input_image)
    
    classify_result = {
        'crop_image': crop_image_list,
        'classify_result': classify_list,
        'confidence': confidence_list
    }
    
    # print("="*100)
    # display(input_image)
    # print(" ")
    # print(f"{convert_cls_for_classification(result[0].probs.top1)} {int(float(result[0].probs.top1conf)*100)}%")
    # print(" ")
    # print("="*100)
    
    return classify_result