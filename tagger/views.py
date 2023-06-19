# views.py
import time

from rest_framework import views, status
from rest_framework.response import Response
from .RuPosTagger import RuPosTagger


class TextProcessingView(views.APIView):
    def post(self, request):
        text = request.body.decode('utf-8')
        if text:
            start_time = time.time()
            tagger = RuPosTagger()
            result = tagger.get_nouns(text)
            end_time = time.time()
            return Response({'result': result,'time':end_time-start_time}, status=status.HTTP_200_OK)
        else:
            return Response({'error': 'Invalid or missing text'}, status=status.HTTP_400_BAD_REQUEST)
