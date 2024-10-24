from ultralytics import YOLO
from setproctitle import *
import glob
import sys
import random
import show_detect as sd
import es_opensearch as eos
import os
import random
import re
from IPython.display import HTML, display
import io
import base64

# 문서의 작성일 및 유형 정보 불러오기
def get_format_savedate(path):
    with open(path, 'r') as fp:
        meta = fp.read()
    
    try:
        fm = re.search(r'Format   : \\[(.+)\\]\\nFormat Code', meta, flags=re.DOTALL).groups()[0]
        date = re.search(r'SaveDTM  : \\[(\\d{4}[/-]\\d{2}[/-]\\d{2}).+\\]', meta, flags=re.DOTALL).groups()[0]
        
    except AttributeError as e:
        return (None,None)
    return fm, date

# 상대경로 자동 계산
def get_relative_path(path):
    working_directory = os.getcwd()
    
    relative_path = os.path.relpath(path, working_directory)
    
    return relative_path

# Yolo 모델 불러오기 - Detection(1차 분류) 및 Classification(2차 분류) 모델
class ModelLoader:
    
    def __init__(self, yolo_model_path, yolo_classify_model_path, device_number='1'):
        self.device_number = device_number
        self.yolo_model_path = yolo_model_path
        self.yolo_classify_model_path = yolo_classify_model_path
        
        self.set_environment()
        self.detection_model = self.load_yolov8_detection_model()
        self.classify_model = self.load_yolov8_classify_model()
    
    # GPU 메모리 번호 할당
    def set_environment(self):
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = self.device_number

    # Detection 모델 로드    
    def load_yolov8_detection_model(self):
        detection_model = YOLO(self.yolo_model_path)
        return detection_model
    
    # Classification 모델 로드
    def load_yolov8_classify_model(self):
        classify_mode = YOLO(self.yolo_classify_model_path)
        return classify_mode
    
# 문서의 이미지 파일 불러오기
class LoadFile:
    # test 데이터셋으로 디렉토리 안의 임의의 데이터 불러오기
    @staticmethod
    def random_file_in_test(test_directory):
        d_list = [d for d in os.listdir(test_directory) if os.path.isdir(os.path.join(test_directory,d))]

        number = random.randint(0,len(d_list)-1)

        random_file_path = test_directory+d_list[number]

        images = glob.glob(random_file_path+'/*.png')
        txt = glob.glob(random_file_path+'/*.txt')

        file_dic = {
            'name': random_file_path,
            'images':images,
            'txt':txt
        }

        return file_dic
    
    # 문서 Lake의 특정 일련번호 문서의 모든 이미지 불러오기
    @staticmethod
    def directory_file(first_dir_number, seconde_dir_number, file_number):
        dir_path = f'/data/lake/ensol/processed/{first_dir_number}/{seconde_dir_number}/'

        png_files = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.startswith(f'{file_number}') and file.endswith('.png'):
                    images = os.path.join(dir_path + file)
                    png_files.append(images)

        file_dic = {
            'images': png_files
        }

        return file_dic
    
    # 하나의 이미지 불러오기
    @staticmethod
    def one_file(file_path):

        file_dic = {
            'images': [file_path]
        }

        return file_dic
    

# 1차 분류(Detection) 및 2차 분류(Classification) 추론
class Inference:
    
    def __init__(self, detection_model, classify_model, test_directory=None):
        self.test_directory = test_directory
        self.model = detection_model
        self.classify_model = classify_model
        
    # ============================================================ Detection ===================================================================    
    # 테스트 데이터셋 Detection
    def inference_test(self):
        r = LoadFile().random_file_in_test(self.test_directory)
        
        inference_result_list = []
        crop_image_list = []
        input_images_list = []
        for i in r['images']:
            inference_result, crop_image, input_image = sd.show_image_bbox(i, self.model)
            
            inference_result_list.append(inference_result)
            crop_image_list.append(crop_image)
            input_images_list.append(input_image)
            
        bboxed_images = {
            'bboxed_images': input_images_list
        }
            
        return r, inference_result_list, crop_image_list, bboxed_images
    
    # 문서 Lake의 특정 일련번호 문서의 이미지 Detection
    def inference_directory_with_eos(self):
        queried_index = int(input("[검색된 문서 id] "))
        queried_doc = eos.search_doc('doc_lake_index', queried_index)

        file_num = queried_doc['_source']['lake']['id'][0:1]
        sec=queried_doc['_source']['lake']['id'][1:3]
        fir=queried_doc['_source']['lake']['id'][3:5]

        d = LoadFile.directory_file(f'{fir}', f'{sec}', f'{file_num}')
        
        inference_result_list = []
        crop_image_list = []
        bboxed_images_list= []
        
        # 검색된 파일명
        file_name = queried_doc['_source']['lake']['name']

        # 검색된 문서명
        doc_name = queried_doc['_source']['origin']['name']

        filtered_path = '/data/lake/ensol/filtered_3/'

        doc_type = queried_doc['_source']['origin']['type']
        fm, date = get_format_savedate(os.path.join(filtered_path,fir,sec, queried_doc['_source']['lake']['id']+doc_type+'-info.txt'))

        print('')
        print('[파일명] ', file_name)
        print('')
        print('[문서명] ', doc_name)
        print('')
        print(f'[유형] {fm}  [작성일] {date}')
        
        
        for i in d['images']:
            if not i.split('.')[0].endswith('checkpoint'):
                inference_result, crop_image, input_image = sd.show_image_bbox_first_step(i, self.model, queried_index)

                inference_result_list.append(inference_result)
                crop_image_list.append(crop_image)
                bboxed_images_list.append(input_image)

        return d, inference_result_list, crop_image_list, bboxed_images_list
    
    # 하나의 이미지 Detection
    def inference_one_file(self, file_path):
        f = LoadFile().one_file(file_path)
        
        inference_result_list = []
        crop_image_list = []
        result_list = []
        for i in f['images']:
            inference_result, crop_image, result = sd.show_image_bbox_first_step(i, self.model)
            
            inference_result_list.append(inference_result)
            crop_image_list.append(crop_image)
            result_list.append(result)
            
        return f, inference_result_list, crop_image_list, result_list
    
    # ============================================================ Classify ===================================================================
    # 2차 분류
    def classify(self, crop_path):
        classify_result_list = []
        for _, line in enumerate(crop_path):
            for path in line['crop_path']:
                A = sd.show_cropped_classified_image(path, self.classify_model)
                classify_result_list.append(A)
                
        return classify_result_list

    
# 이미지 식별/분류 전체 결과 시각화
class Visualize:
    # 문서의 페이지별 메타 정보, 원본 이미지, crop 이미지 및 분류 결과 시각화
    @staticmethod
    def multi_image_inference_visualize(file_dic_image, inference_result_list, crop_image_list, classify_result_list, bboxed_images):
        
        # 문서의 페이지별 메타 정보
        for idx, image_path in enumerate(file_dic_image['images']):
            image_file_name = image_path.split('/')[-1]
            serial_number = image_file_name[:5]
            doc_type = image_file_name.split('.')[-1]
            inference_result = [f"{key} {value}개" for key, value in inference_result_list[idx].items()]
            
            title_html = f"""<table style="border-collapse: collapse; width: 100%;">    
    <tr>     <th style="border: 1px solid #dddddd; text-align: left; padding: 8px; margin: 0px; text-align: center">파일명</th>
            <th style="border: 1px solid #dddddd; text-align: left; padding: 8px; text-align: center">일련번호</th>
            <th style="border: 1px solid #dddddd; text-align: left; padding: 8px; text-align: center">추론 결과</th>
            <th style="border: 1px solid #dddddd; text-align: left; padding: 8px; text-align: center">문서유형</th>

    </tr> 
    <tr>
    <td style="border: 1px solid #dddddd; text-align: left; padding: 8px; font-weight: bold; text-align: center">{image_file_name}</td>
    <td style="border: 1px solid #dddddd; text-align: left; padding: 8px; font-weight: bold; text-align: center">{serial_number}</td>
    <td style="border: 1px solid #dddddd; text-align: left; padding: 8px; font-weight: bold; text-align: center">{inference_result}</td>
    <td style="border: 1px solid #dddddd; text-align: left; padding: 8px; font-weight: bold; text-align: center">{f'{doc_type}'}</td>

    </tr></table>"""
            
            # 원본 이미지
            img = bboxed_images[idx]
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            image_html = f"""<table style="border-collapse: collapse; width: 100%;">    
        <tr>     
        <th style="border: 1px solid #dddddd; text-align: left; padding: 8px;text-align: center">Image</th>     
        </tr>
        <tr>
            <td style="border: 1px solid #dddddd; text-align: left; padding: 8px;">    <img src="data:image/jpeg;base64,{img_str}" alt="이미지" style="max-width: 50%; height: auto; display: block; margin: 0 auto">     </td> 
        </tr>
        """
        
            html_code = f"""<table style="border-collapse: collapse; width: 100%;">    
    <tr>     <th style="border: 1px solid #dddddd; text-align: left; padding: 8px; margin: 0px; width: 5%; text-align: center">Page</th>
            <th style="border: 1px solid #dddddd; text-align: left; padding: 8px; width: 5%; text-align: center">추론 결과</th>
            <th style="border: 1px solid #dddddd; text-align: left; padding: 8px;text-align: center">핵심 영역</th>          
    </tr> 
    """     
            # crop 이미지들 및 분류 결과
            for idx2, crop_path in enumerate(crop_image_list[idx]['crop_path']):
                croped_image = get_relative_path(crop_path)
                page = idx2
                for classify_result in classify_result_list:
                    if crop_path == classify_result['crop_image'][0]:
                        crop_image_result = classify_result['classify_result'][0]
                        confidence_score = int(float(classify_result['confidence'][0])*100)
                
                        html_code += f'''
                    <tr>
                <td style="border: 1px solid #dddddd; text-align: left; padding: 8px; font-weight: bold; text-align: center">{page}</td> 
                <td style="border: 1px solid #dddddd; text-align: left; padding: 8px; font-weight: bold;text-align: center; font-size: 10pt">{crop_image_result} {confidence_score}%</td>
                <td style="border: 1px solid #dddddd; text-align: left; padding: 8px;">    <img src="{croped_image}" alt="이미지" style="max-width: auto; height: auto;display: block; margin: 0 auto">     </td>
                </tr>
                    '''
                
            display(HTML(title_html))
            display(HTML(image_html))
            display(HTML(html_code))
            print('')
            print('=================================================================================================================================================================================')
            print('')
    

    #=============================================================================================================================== 
    # 하나의 이미지에 대한 crop 이미지와 분류 결과만 시각화
    @staticmethod
    def only_croped_drawing_visualize(crop_image_list, classify_result_list):
        '''
        ########################################################################################
        crop_image_list => 하나의 원본 이미지에서 나온 crop 이미지들의 모음
            
            crop_image_list = [{
                'crop_path' : [crop 이미지들의 경로가 모여있음]
            }]
        ########################################################################################    
        classify_result_list => 각 crop 이미지들에 대해 classify한 결과들 모음
        
            classify_result_list = [{
                'crop_image' : [crop 이미지 하나의 경로, crop_image_list의 'crop_path' 와 일치함],
                'classify_result' : [crop 이미지 하나의 분류 결과]
                'confidence' : [분류의 confidence값]
            }, ...]
        ########################################################################################
        '''
        
        
            
        html_code = f"""<table style="border-collapse: collapse; width: 100%;">    
<tr>     <th style="border: 1px solid #dddddd; text-align: left; padding: 8px; margin: 0px; width: 5%; text-align: center">Page</th>
        <th style="border: 1px solid #dddddd; text-align: left; padding: 8px; width: 5%; text-align: center">추론 결과</th>
        <th style="border: 1px solid #dddddd; text-align: left; padding: 8px;text-align: center">핵심 영역</th>          
</tr> 
"""
        for idx, crop_path in enumerate(crop_image_list[0]['crop_path']):
            croped_image = get_relative_path(crop_path)
            page = idx
            for classify_result in classify_result_list:
                if crop_path == classify_result['crop_image'][0]:
                    crop_image_result = classify_result['classify_result'][0]
                    confidence_score = int(float(classify_result['confidence'][0])*100)

                    html_code += f'''
                <tr>
            <td style="border: 1px solid #dddddd; text-align: left; padding: 8px; font-weight: bold; text-align: center">{page}</td> 
            <td style="border: 1px solid #dddddd; text-align: left; padding: 8px; font-weight: bold;text-align: center; font-size: 10pt">{crop_image_result} {confidence_score}%</td>
            <td style="border: 1px solid #dddddd; text-align: left; padding: 8px;">    <img src="{croped_image}" alt="이미지" style="max-width: auto; height: auto;display: block; margin: 0 auto">     </td>
            </tr>
                '''

        display(HTML(html_code))