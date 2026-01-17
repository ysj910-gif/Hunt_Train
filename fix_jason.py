import json
import os

# 1. 파일 경로 설정 (본인의 파일명으로 수정하세요)
file_path = 'map_data.json' 

if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 발판(platforms) 데이터에 id가 없으면 추가
    for i, platform in enumerate(data.get('platforms', [])):
        if 'id' not in platform:
            platform['id'] = i  # 0, 1, 2... 순서대로 부여

    # 3. 파일 다시 저장
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"성공: {len(data['platforms'])}개의 발판에 ID를 부여했습니다!")
else:
    print("파일을 찾을 수 없습니다. 경로를 확인해주세요.")