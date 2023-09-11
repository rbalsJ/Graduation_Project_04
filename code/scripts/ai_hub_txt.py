# import os
# import json

# # 작업할 폴더 경로
# folder_path = "data/raw/AI_hub"  # 폴더 경로를 적절히 수정해주세요.

# # 폴더 내의 모든 JSON 파일 찾기
# def find_json_files(folder_path):
#     json_files = []
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             if file.endswith(".json"):
#                 json_files.append(os.path.join(root, file))
#     return json_files

# json_files = find_json_files(folder_path)

# # 중복 없이 모든 convrsThema 값을 저장할 set
# unique_convrsThema = set()

# # 각 JSON 파일에 대해 작업 수행
# for json_file_path in json_files:
#     with open(json_file_path, "r", encoding="utf-8") as file:  # 인코딩을 utf-8로 변경
#         data = json.load(file)
        
#         convrsThema = data["대화정보"]["convrsThema"]
        
#         # 중복 없이 값 저장
#         unique_convrsThema.add(convrsThema)

# # 중복 없는 값들을 리스트로 변환하고 사전순으로 정렬
# sorted_unique_convrsThema = sorted(list(unique_convrsThema))

# # 모든 convrsThema 값을 저장할 하나의 txt 파일 경로
# output_txt_file_path = os.path.join('code/scripts', "sorted_unique_convrsThema.txt")

# with open(output_txt_file_path, "w", encoding="utf-8") as output_txt_file:
#     # 정렬된 값들을 txt 파일에 저장
#     for convrsThema in sorted_unique_convrsThema:
#         output_txt_file.write(convrsThema + '\n')

# print("작업이 완료되었습니다.")


import os
import json

# 작업할 폴더 경로
folder_path = "data/raw/AI_hub"  # 폴더 경로를 적절히 수정해주세요.

# 폴더 내의 모든 JSON 파일 찾기
def find_json_files(folder_path):
    json_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files

json_files = find_json_files(folder_path)

# 모든 convrsThema 값을 저장할 하나의 txt 파일 경로
output_txt_file_path = os.path.join('code/scripts', "combined_convrsThema.txt")

with open(output_txt_file_path, "w", encoding="utf-8") as output_txt_file:
    # 각 JSON 파일에 대해 작업 수행
    for json_file_path in json_files:
        with open(json_file_path, "r", encoding="utf-8") as file:  # 인코딩을 utf-8로 변경
            data = json.load(file)
            
            convrsThema = data["대화정보"]["convrsThema"]
            
            # convrsThema 값을 txt 파일에 추가
            output_txt_file.write(convrsThema + '\n')

print("작업이 완료되었습니다.")