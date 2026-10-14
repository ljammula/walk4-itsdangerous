import json
from datetime import timedelta
from functools import partial

import pytest

from itsdangerous.exc import BadSignature
from itsdangerous.exc import SignatureExpired
from itsdangerous.url_safe import URLSafeSerializer
from itsdangerous.url_safe import URLSafeTimedSerializer
from test_itsdangerous.test_serializer import KeywordSerializer
from test_itsdangerous.test_serializer import TestSerializer
from test_itsdangerous.test_timed import TestTimedSerializer


class TestURLSafeSerializer(TestSerializer):
    @pytest.fixture()
    def serializer_factory(self):
        return partial(URLSafeSerializer, secret_key="secret-key")

    @pytest.fixture(params=({"id": 42}, pytest.param("a" * 1000, id="zlib")))
    def value(self, request):
        return request.param

    @pytest.mark.parametrize(
        ("value", "compressed"),
        (
            pytest.param({"id": 42}, False, id="uncompressed"),
            pytest.param("a" * 1000, True, id="compressed"),
        ),
    )
    def test_serializer_kwargs_load(
        self, serializer_factory, value, compressed
    ):
        data_serializer = KeywordSerializer()
        serializer = serializer_factory(
            serializer=data_serializer,
            serializer_kwargs={"required": "configured"},
        )
        signed = serializer.dumps(value)

        assert signed.startswith(".") is compressed
        assert serializer.loads(signed) == value
        payload, kwargs = data_serializer.loads_calls[-1]
        assert payload == json.dumps(value)
        assert kwargs == {"required": "configured"}

        with pytest.raises(BadSignature):
            serializer.loads(signed[:-1])


class TestURLSafeTimedSerializer(TestURLSafeSerializer, TestTimedSerializer):
    @pytest.fixture()
    def serializer_factory(self):
        return partial(URLSafeTimedSerializer, secret_key="secret-key")

    def test_serializer_kwargs_load_timed(self, serializer_factory, ts, freeze):
        data_serializer = KeywordSerializer()
        serializer = serializer_factory(
            serializer=data_serializer,
            serializer_kwargs={"required": "configured"},
        )
        value = "a" * 1000
        signed = serializer.dumps(value)

        loaded, timestamp = serializer.loads(
            signed, max_age=10, return_timestamp=True
        )
        assert loaded == value
        assert timestamp == ts
        assert data_serializer.loads_calls[-1] == (
            json.dumps(value),
            {"required": "configured"},
        )

        with pytest.raises(BadSignature):
            serializer.loads(signed[:-1], max_age=10)

        freeze.tick(timedelta(seconds=11))
        with pytest.raises(SignatureExpired):
            serializer.loads(signed, max_age=10)
