from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

path_img = '/Users/dariatunina/mach-lerinig/single_digits/'
size = 40, 40
font = ImageFont.truetype("fonts/COMIC_SANS.ttf", 20)

for i in range(0, 10):
    img = Image.new('RGB', (20, 20), color=(255, 255, 255))
    #Image.open("help_stuff/blank_page2.jpg")
    img.thumbnail(size, Image.ANTIALIAS)
    draw = ImageDraw.Draw(img)
    print(str(i) + ': ' + str(font.getsize(str(i))))
    draw.text(xy=(0, 0), text=str(i), fill=(0, 0, 0), font=font)
    img.save(path_img + str(i) + '.jpg')
print('" ": ' + str(font.getsize(' ')))
print('+: ' + str(font.getsize('+')))
print('-: ' + str(font.getsize('-')))
print('=: ' + str(font.getsize('=')))


