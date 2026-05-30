from pymc_core import LocalIdentity
from pymc_core.protocol import CryptoUtils
from pymc_core.protocol.constants import (
    MAX_PACKET_PAYLOAD,
    PAYLOAD_TYPE_ACK,
    PAYLOAD_TYPE_ADVERT,
    PAYLOAD_TYPE_ANON_REQ,
    PAYLOAD_TYPE_PATH,
    PAYLOAD_TYPE_RAW_CUSTOM,
)
from pymc_core.protocol.identity import Identity
from pymc_core.protocol.packet import Packet
from pymc_core.protocol.packet_builder import PacketBuilder
from pymc_core.protocol.packet_utils import PathUtils


# PacketBuilder tests
def test_packet_builder_create_ack():
    """Test creating ACK packets."""
    identity = LocalIdentity()
    timestamp = 1234567890
    attempt = 1
    text = "test_ack"

    ack_packet = PacketBuilder.create_ack(identity.get_public_key(), timestamp, attempt, text)

    assert ack_packet is not None
    assert ack_packet.get_payload_type() == PAYLOAD_TYPE_ACK


def test_packet_builder_create_advert():
    """Test creating advertisement packets."""
    identity = LocalIdentity()
    advert_packet = PacketBuilder.create_advert(identity, "test_data", 1)

    assert advert_packet is not None
    assert advert_packet.get_payload_type() == PAYLOAD_TYPE_ADVERT


def test_packet_builder_create_self_advert():
    """Test creating self-advertisement packets."""
    identity = LocalIdentity()
    self_advert = PacketBuilder.create_self_advert(identity, "TestNode", 1)

    assert self_advert is not None
    assert self_advert.get_payload_type() == PAYLOAD_TYPE_ADVERT


def test_packet_builder_create_flood_advert():
    """Test creating flood advertisement packets."""
    identity = LocalIdentity()
    flood_advert = PacketBuilder.create_flood_advert(identity, "TestNode", 1)

    assert flood_advert is not None
    assert flood_advert.get_payload_type() == PAYLOAD_TYPE_ADVERT


def test_packet_builder_create_direct_advert():
    """Test creating direct advertisement packets."""
    identity = LocalIdentity()
    direct_advert = PacketBuilder.create_direct_advert(identity, "TestNode", 1)

    assert direct_advert is not None
    assert direct_advert.get_payload_type() == PAYLOAD_TYPE_ADVERT


def test_packet_builder_create_raw_data():
    """Test creating raw custom packets (PAYLOAD_TYPE_RAW_CUSTOM)."""
    data = b"\x01\x02\x03\x04"
    pkt = PacketBuilder.create_raw_data(data)
    assert pkt is not None
    assert pkt.get_payload_type() == PAYLOAD_TYPE_RAW_CUSTOM
    assert pkt.payload == bytearray(data)
    assert pkt.payload_len == len(data)
    assert pkt.path_len == 0
    assert pkt.path == bytearray()


def test_packet_builder_create_raw_data_too_large_raises():
    """Test create_raw_data raises when data exceeds MAX_PACKET_PAYLOAD."""
    import pytest

    data = bytes(MAX_PACKET_PAYLOAD + 1)
    with pytest.raises(ValueError, match="exceeds MAX_PACKET_PAYLOAD"):
        PacketBuilder.create_raw_data(data)


def test_packet_builder_create_path_return_encoded_path_len():
    """Inner payload first byte must be encoded path_len (hash size + hop count),
    not path byte count.

    With 2-byte hashes and 2 hops, path is 4 bytes. Encoded path_len = 0x42.
    Without path_len_encoded, first byte would be 4 (wrong: decoded as 4 hops × 1-byte).
    """
    path_len_encoded = PathUtils.encode_path_len(2, 2)  # 0x42: 2-byte hashes, 2 hops
    assert path_len_encoded == 0x42
    path_byte_len = PathUtils.get_path_byte_len(path_len_encoded)
    assert path_byte_len == 4

    path = list(bytes(range(4)))  # 4 path bytes
    secret = bytes(32)  # 32-byte shared secret
    pkt = PacketBuilder.create_path_return(
        dest_hash=0xAB,
        src_hash=0xCD,
        secret=secret,
        path=path,
        extra_type=0xFF,
        extra=b"",
        path_len_encoded=path_len_encoded,
    )
    assert pkt.get_payload_type() == PAYLOAD_TYPE_PATH
    assert pkt.payload[0] == 0xAB
    assert pkt.payload[1] == 0xCD

    aes_key = secret[:16]
    cipher = bytes(pkt.payload[2:])
    decrypted = CryptoUtils.mac_then_decrypt(aes_key, secret, cipher)
    assert decrypted[0] == 0x42, "first byte must be encoded path_len 0x42, not path byte count 4"
    assert PathUtils.get_path_hash_size(decrypted[0]) == 2
    assert PathUtils.get_path_hash_count(decrypted[0]) == 2
    assert decrypted[1:5] == bytes(path)
    assert decrypted[5] == 0xFF  # extra_type


def test_packet_builder_create_path_return_no_encoded_uses_len_path():
    """When path_len_encoded is None, first byte is len(path) (1-byte hash semantics)."""
    path = [0x11, 0x22, 0x33]  # 3 bytes
    secret = bytes(32)
    pkt = PacketBuilder.create_path_return(
        dest_hash=0x01,
        src_hash=0x02,
        secret=secret,
        path=path,
        extra_type=0xFF,
        extra=b"",
        path_len_encoded=None,
    )
    aes_key = secret[:16]
    decrypted = CryptoUtils.mac_then_decrypt(aes_key, secret, bytes(pkt.payload[2:]))
    assert decrypted[0] == 3
    assert decrypted[1:4] == bytes(path)


def test_create_text_message_cli_data_flags_byte():
    """TXT_TYPE_CLI_DATA sets upper bits of flags; ACK crc includes full flags byte."""
    local = LocalIdentity()
    other = LocalIdentity()
    contact = type(
        "Contact",
        (),
        {
            "public_key": other.get_public_key().hex(),
            "out_path": [],
            "out_path_len": -1,
        },
    )()
    pkt_plain, crc_plain = PacketBuilder.create_text_message(
        contact, local, "cmd", 1, "direct", None, 0
    )
    pkt_cli, crc_cli = PacketBuilder.create_text_message(
        contact, local, "cmd", 1, "direct", None, 1
    )
    peer_pub = local.get_public_key()
    secret = Identity(peer_pub).calc_shared_secret(other.get_private_key())
    aes_key = secret[:16]

    def _dec_txt(p):
        return CryptoUtils.mac_then_decrypt(aes_key, secret, bytes(p.payload[2:]))

    dec_p = _dec_txt(pkt_plain)
    dec_c = _dec_txt(pkt_cli)
    assert dec_p[4] == 0x01  # PLAIN: (0 << 2) | attempt 1
    assert dec_c[4] == 0x05  # CLI_DATA: (1 << 2) | attempt 1
    assert crc_plain != crc_cli


def test_create_text_message_truncated_path_path_len_consistency():
    """When contact has 64-byte path but out_path_len encodes more than 64 bytes
    (e.g. 33 hops × 2-byte = 66), do not use contact_path_len; use 1-byte
    encoding and cap path at 63 so path_len never declares more bytes than present.
    """
    local = LocalIdentity()
    other = LocalIdentity()
    # 64-byte path, but encoded as 33 hops × 2-byte = 66 (invalid to use)
    contact_path_len_66 = PathUtils.encode_path_len(2, 33)  # 0x61, 66 bytes
    assert PathUtils.get_path_byte_len(contact_path_len_66) == 66
    contact = type(
        "Contact",
        (),
        {
            "public_key": other.get_public_key().hex(),
            "out_path": list(range(64)),
            "out_path_len": contact_path_len_66,
        },
    )()
    pkt, _ = PacketBuilder.create_text_message(contact, local, "hi", 0, "direct", out_path=None)
    # Must not have used contact_path_len (66 > 64); path should be 63 bytes, 1-byte encoding
    assert pkt.get_path_byte_len() <= len(pkt.path)
    assert pkt.get_path_byte_len() == 63
    assert len(pkt.path) == 63


def test_create_protocol_request_truncated_path_path_len_consistency():
    """When contact out_path is > 64 bytes and out_path_len encodes > 64 bytes,
    truncate path and do not use out_path_len; cap at 63 and use 1-byte encoding.
    """
    local = LocalIdentity()
    other = LocalIdentity()
    out_path_len_66 = PathUtils.encode_path_len(2, 33)  # 66 bytes
    contact = type(
        "Contact",
        (),
        {
            "public_key": other.get_public_key().hex(),
            "out_path": bytes(range(70)),
            "out_path_len": out_path_len_66,
        },
    )()
    packet, _ = PacketBuilder.create_protocol_request(contact, local, 0x01, b"")
    assert packet.get_path_byte_len() <= len(packet.path)
    assert packet.get_path_byte_len() == 63
    assert len(packet.path) == 63


def test_create_login_packet_truncated_path_path_len_consistency():
    """Same as create_protocol_request: truncated path must not use out_path_len
    when it would imply more bytes than present.
    """
    local = LocalIdentity()
    other = LocalIdentity()
    out_path_len_66 = PathUtils.encode_path_len(2, 33)
    contact = type(
        "Contact",
        (),
        {
            "public_key": other.get_public_key().hex(),
            "out_path": bytes(range(70)),
            "out_path_len": out_path_len_66,
        },
    )()
    pkt = PacketBuilder.create_login_packet(contact, local, "secret")
    assert pkt.get_path_byte_len() <= len(pkt.path)
    assert pkt.get_path_byte_len() == 63
    assert len(pkt.path) == 63


def test_truncated_path_packet_round_trip():
    """Packet built with truncated path and safe path_len must write_to/read_from
    without error and without 'truncated path'."""
    local = LocalIdentity()
    other = LocalIdentity()
    out_path_len_66 = PathUtils.encode_path_len(2, 33)
    contact = type(
        "Contact",
        (),
        {
            "public_key": other.get_public_key().hex(),
            "out_path": bytes(range(70)),
            "out_path_len": out_path_len_66,
        },
    )()
    packet, _ = PacketBuilder.create_protocol_request(contact, local, 0x01, b"data")
    raw = packet.write_to()
    pkt2 = Packet()
    ok = pkt2.read_from(raw)
    assert ok
    assert pkt2.get_path_byte_len() == len(pkt2.path)
    assert pkt2.get_path_byte_len() == 63


def _make_contact(other, out_path=b"", out_path_len=-1):
    return type(
        "Contact",
        (),
        {
            "public_key": other.get_public_key().hex(),
            "out_path": out_path,
            "out_path_len": out_path_len,
        },
    )()


def _decrypt_anon(pkt, sender_local, recipient_local):
    """Decrypt an ANON_REQ packet: payload = dest_hash(1)+sender_pubkey(32)+cipher."""
    assert pkt.payload[1:33] == bytes(sender_local.get_public_key())
    cipher = bytes(pkt.payload[33:])
    secret = Identity(sender_local.get_public_key()).calc_shared_secret(
        recipient_local.get_private_key()
    )
    return CryptoUtils.mac_then_decrypt(secret[:16], secret, cipher)


def test_create_anon_request_is_anon_payload_type_no_subtype_prefix():
    """Regression: anon requests must be PAYLOAD_TYPE_ANON_REQ with the client's
    sub-type byte at offset 4 (after the 4-byte timestamp) - NOT a PAYLOAD_TYPE_REQ
    with 0x07 prepended (which repeaters read as REQ_TYPE_GET_OWNER_INFO)."""
    local = LocalIdentity()
    other = LocalIdentity()
    contact = _make_contact(other, out_path=b"", out_path_len=0)  # zero-hop neighbour
    # ANON_REQ_TYPE_REGIONS (0x01) + reply-path byte (0 = empty path)
    req_data = bytes([0x01, 0x00])
    pkt, ts = PacketBuilder.create_anon_request(contact, local, req_data)

    assert pkt.get_payload_type() == PAYLOAD_TYPE_ANON_REQ
    plaintext = _decrypt_anon(pkt, local, other)
    assert int.from_bytes(plaintext[:4], "little") == ts
    # sub-type byte sits immediately after the timestamp, with no 0x07 prefix
    assert plaintext[4] == 0x01
    # (trailing bytes are AES block padding, ignored by the responder)
    assert plaintext[4 : 4 + len(req_data)] == req_data


def test_create_anon_request_zero_hop_is_direct():
    """out_path_len == 0 (zero-hop direct neighbour) must route DIRECT so the
    firmware regions handler (which requires isRouteDirect()) answers."""
    local = LocalIdentity()
    other = LocalIdentity()
    contact = _make_contact(other, out_path=b"", out_path_len=0)
    pkt, _ = PacketBuilder.create_anon_request(contact, local, bytes([0x01, 0x00]))
    assert pkt.is_route_direct()
    assert not pkt.is_route_flood()


def test_create_anon_request_unknown_path_is_flood():
    """out_path_len == -1 (unknown) must route FLOOD."""
    local = LocalIdentity()
    other = LocalIdentity()
    contact = _make_contact(other, out_path=b"", out_path_len=-1)
    pkt, _ = PacketBuilder.create_anon_request(contact, local, bytes([0x01, 0x00]))
    assert pkt.is_route_flood()
