use rustdds::{RepresentationIdentifier, dds::adapters::no_key};

/// Raw CDR payload bytes including the 4-byte encapsulation header.
///
/// Used with [`RawBytesAdapter`] to capture DDS messages without any CDR
/// struct interpretation, making the payload compatible with the existing
/// `CdrDeserializer` in `rosbag::cdr`.
#[derive(Debug, Clone)]
pub struct RawBytes(pub Vec<u8>);

/// Decoder that copies incoming bytes and prepends the CDR header.
///
/// rustdds strips the 4-byte CDR encapsulation header before calling the
/// decoder, passing the encoding separately. We reconstruct the full payload
/// by prepending the header so `CdrDeserializer::new()` (which expects
/// `data[0..4]` to be the header) can process the data without modification.
#[derive(Clone)]
pub struct RawBytesDecoder;

impl no_key::Decode<RawBytes> for RawBytesDecoder {
    type Error = std::convert::Infallible;

    fn decode_bytes(
        self,
        input_bytes: &[u8],
        encoding: RepresentationIdentifier,
    ) -> Result<RawBytes, Self::Error> {
        // Reconstruct the 4-byte CDR encapsulation header from the encoding.
        let header = match encoding {
            RepresentationIdentifier::CDR_LE | RepresentationIdentifier::PL_CDR_LE => {
                [0x00u8, 0x01, 0x00, 0x00]
            }
            _ => [0x00u8, 0x00, 0x00, 0x00], // CDR_BE default
        };
        let mut data = Vec::with_capacity(4 + input_bytes.len());
        data.extend_from_slice(&header);
        data.extend_from_slice(input_bytes);
        Ok(RawBytes(data))
    }
}

const SUPPORTED: &[RepresentationIdentifier] = &[
    RepresentationIdentifier::CDR_BE,
    RepresentationIdentifier::CDR_LE,
    RepresentationIdentifier::PL_CDR_BE,
    RepresentationIdentifier::PL_CDR_LE,
];

/// DeserializerAdapter that captures raw CDR bytes without any serde parsing.
pub struct RawBytesAdapter;

impl no_key::DeserializerAdapter<RawBytes> for RawBytesAdapter {
    type Error = std::convert::Infallible;
    type Decoded = RawBytes;

    fn supported_encodings() -> &'static [RepresentationIdentifier] {
        SUPPORTED
    }

    fn transform_decoded(decoded: Self::Decoded) -> RawBytes {
        decoded
    }
}

impl no_key::DefaultDecoder<RawBytes> for RawBytesAdapter {
    type Decoder = RawBytesDecoder;
    const DECODER: Self::Decoder = RawBytesDecoder;
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use rustdds::dds::adapters::no_key::Decode;

    #[test]
    fn decode_cdr_le_prepends_le_header() {
        let payload = b"hello";
        let decoded = RawBytesDecoder
            .decode_bytes(payload, RepresentationIdentifier::CDR_LE)
            .unwrap();
        assert_eq!(&decoded.0[..4], &[0x00, 0x01, 0x00, 0x00]);
        assert_eq!(&decoded.0[4..], payload);
    }

    #[test]
    fn decode_cdr_be_prepends_be_header() {
        let payload = b"world";
        let decoded = RawBytesDecoder
            .decode_bytes(payload, RepresentationIdentifier::CDR_BE)
            .unwrap();
        assert_eq!(&decoded.0[..4], &[0x00, 0x00, 0x00, 0x00]);
        assert_eq!(&decoded.0[4..], payload);
    }

    #[test]
    fn decode_pl_cdr_le_prepends_le_header() {
        let payload = b"test";
        let decoded = RawBytesDecoder
            .decode_bytes(payload, RepresentationIdentifier::PL_CDR_LE)
            .unwrap();
        assert_eq!(&decoded.0[..4], &[0x00, 0x01, 0x00, 0x00]);
    }

    #[test]
    fn decode_empty_payload_produces_only_header() {
        let decoded = RawBytesDecoder
            .decode_bytes(&[], RepresentationIdentifier::CDR_LE)
            .unwrap();
        assert_eq!(decoded.0.len(), 4);
    }

    #[test]
    fn supported_encodings_contains_all_four() {
        let supported =
            <RawBytesAdapter as no_key::DeserializerAdapter<RawBytes>>::supported_encodings();
        assert!(supported.contains(&RepresentationIdentifier::CDR_LE));
        assert!(supported.contains(&RepresentationIdentifier::CDR_BE));
        assert!(supported.contains(&RepresentationIdentifier::PL_CDR_LE));
        assert!(supported.contains(&RepresentationIdentifier::PL_CDR_BE));
    }

    #[test]
    fn transform_decoded_is_identity() {
        let raw = RawBytes(vec![1, 2, 3]);
        let out = <RawBytesAdapter as no_key::DeserializerAdapter<RawBytes>>::transform_decoded(
            raw.clone(),
        );
        assert_eq!(out.0, raw.0);
    }
}
