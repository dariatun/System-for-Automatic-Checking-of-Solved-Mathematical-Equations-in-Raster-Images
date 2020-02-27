from tesserocr import PyTessBaseAPI
api = PyTessBaseAPI(path='/datagrid/personal/tunindar/tensorflow/', lang='eng')
try:
        api.SetImageFile('7104_3_4413.jpg')
        print(api.GetUTF8Text())
finally:
        api.End()
