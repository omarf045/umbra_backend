from django.shortcuts import render

from rest_framework.response import Response
from .serializers import UserSerializer, RegisterSerializer
from django.contrib.auth.models import User
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser

from rest_framework.renderers import JSONRenderer
from rest_framework.authtoken.models import Token
from rest_framework.authtoken.serializers import AuthTokenSerializer

from django.contrib.auth.tokens import PasswordResetTokenGenerator
from rest_framework.compat import coreapi, coreschema
from rest_framework.schemas import coreapi as coreapi_schema
from rest_framework.schemas import ManualSchema

from rest_framework import generics

from rest_framework.views import APIView

import base64
import io
from .utils import get_protected_image, get_phone_number

import threading
import base64

from email.message import EmailMessage
import smtplib

import datetime

class UserDetailAPI(APIView):
    authentication_classes = (TokenAuthentication,)
    permission_classes = (AllowAny,)

    def get(self, request, *args, **kwargs):
        user = User.objects.get(id=request.user.id)
        serializer = UserSerializer(user)
        return Response(serializer.data)

class RegisterUserAPI(generics.CreateAPIView):
    permission_classes = (AllowAny,)
    serializer_class = RegisterSerializer

class LoginUserAPI(generics.CreateAPIView):
    parser_classes = (MultiPartParser, FormParser, JSONParser)
    renderer_classes = (JSONRenderer,)
    serializer_class = AuthTokenSerializer

    if coreapi_schema.is_enabled():
        schema = ManualSchema(
            fields=[
                coreapi.Field(
                    name="username",
                    required=True,
                    location='form',
                    schema=coreschema.String(
                        title="Username",
                        description="Valid username for authentication",
                    ),
                ),
                coreapi.Field(
                    name="password",
                    required=True,
                    location='form',
                    schema=coreschema.String(
                        title="Password",
                        description="Valid password for authentication",
                    ),
                ),
            ],
            encoding="application/json",
        )

    def get_serializer_context(self):
        return {
            'request': self.request,
            'format': self.format_kwarg,
            'view': self
        }

    def get_serializer(self, *args, **kwargs):
        kwargs['context'] = self.get_serializer_context()
        return self.serializer_class(*args, **kwargs)

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        token, created = Token.objects.get_or_create(user=user)
        now = datetime.datetime.now()
        user.last_login = str(now)
        user.save()
        return Response({'token': token.key})

class VerifyUserAPI(generics.CreateAPIView):
    permission_classes = (AllowAny,)

    def post(self, request, *args, **kwargs):

        raw_token = request.data['token']
        decoded_token = raw_token.replace("%3D", "=")
        token_array = decoded_token.split("%2F")

        uidb64 = token_array[0]
        token = token_array[1]

        uidb64_bytes = uidb64.encode('ascii')
        uid_bytes = base64.b64decode(uidb64_bytes)
        uid = int(uid_bytes.decode('ascii'))

        user = User.objects.get(id=uid)

        print(uid, token)

        token_generator = PasswordResetTokenGenerator()

        print(user.is_active)

        if not user.is_active:
            if token_generator.check_token(user, token):
                user.is_active = True
                user.save()
                return Response("User verified successfully")
            else:
                return Response("Tokens don't match")
        else:
            return Response("The user is already verified")

class PasswordResetAPI(generics.CreateAPIView):

    def get(self, request, *args, **kwargs):

        def send(user, email_to, token):
            email = EmailMessage()
            email["From"] = settings.EMAIL_HOST_USER
            email["To"] = email_to
            email["Subject"] = "Verificacion de cuenta ALKNOS"

            uid_bytes = str(user.id).encode('ascii')
            uidb64_bytes = base64.b64encode(uid_bytes)
            uidb64 = uidb64_bytes.decode('ascii')

            content = "UIDB64: " + uidb64 + "   TOKEN: " + token

            email.set_content(content)

            smtp = smtplib.SMTP(settings.EMAIL_HOST,
                                port=settings.EMAIL_PORT)
            smtp.starttls()
            smtp.login(settings.EMAIL_HOST_USER, settings.EMAIL_HOST_PASSWORD)
            smtp.sendmail(settings.EMAIL_HOST_USER,
                          email_to, email.as_string())
            smtp.quit()
            print("--- Email has been sent")

        username = request.data['username']
        try:
            user = User.objects.get(username=username)
        except:
            return Response('User does not exist')

        email = user.email

        token_generator = PasswordResetTokenGenerator()

        token_generator = PasswordResetTokenGenerator()
        pwd_reset_token = token_generator.make_token(user)

        print(pwd_reset_token)

        thread = threading.Thread(
            target=send, args=(user, email, pwd_reset_token))
        thread.start()

        return Response('Token has been sent')

    def post(self, request, *args, **kwargs):

        uidb64 = request.data['uid']
        token = request.data['token']

        old_pwd = request.data['old_pwd']
        new_pwd = request.data['new_pwd']

        uidb64_bytes = uidb64.encode('ascii')
        uid_bytes = base64.b64decode(uidb64_bytes)
        uid = uid_bytes.decode('ascii')

        user = User.objects.get(id=int(uid))

        token_generator = PasswordResetTokenGenerator()

        if user.check_password(old_pwd):
            if token_generator.check_token(user, token):
                user.set_password(new_pwd)
                user.save()
                return Response("Password changed successfully")
            else:
                return Response("Tokens don't match")
        else:
            return Response("Incorrect password")

class ProtectImageAPI(APIView):
    authentication_classes = (TokenAuthentication,)
    permission_classes = (IsAuthenticated,)

    def post(self, request, *args, **kwargs):
        phone_number = request.data['phone_number']
        file_base64 = request.data['base64']
        file_base64 = re.sub(r'^.*?base64,', '', file_base64)

        decoded_data = base64.b64decode(file_base64)
        np_data = np.fromstring(decoded_data, np.uint8)
        imagen = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)

        protected_image = get_protected_image(imagen, phone_number)

        _, buffer = cv2.imencode('.jpg', imagen_cv2)
        imagen_base64 = str(base64.b64encode(buffer).decode('utf-8'))

        return Response({"base64": imagen_base64})

class GetImageMarkAPI(APIView):
    authentication_classes = (TokenAuthentication,)
    permission_classes = (IsAuthenticated,)

    def post(self, request, *args, **kwargs):
        file_base64 = request.data['base64']
        file_base64 = re.sub(r'^.*?base64,', '', file_base64)

        decoded_data = base64.b64decode(file_base64)
        np_data = np.fromstring(decoded_data, np.uint8)
        imagen = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)

        phone_number = get_phone_number(imagen)

        return Response({"phone_number": phone_number})