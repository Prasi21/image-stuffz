import re

# https://upload.wikimedia.org/wikipedia/commons/thumb/6/61/Australia_road_sign_R2-4.svg/120px-Australia_road_sign_R2-4.svg.png
# https://upload.wikimedia.org/wikipedia/commons/thumb/6/61/Australia_road_sign_R2-4.svg/480px-Australia_road_sign_R2-4.svg.png
# https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/CA-ON_road_sign_Rb-019.svg/120px-CA-ON_road_sign_Rb-019.svg.pngx

str1 = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/61/Australia_road_sign_R2-4.svg/120px-Australia_road_sign_R2-4.svg.png"

print(str1)

str2 = re.sub("\d+px", "480px", str1)

print(str2)
