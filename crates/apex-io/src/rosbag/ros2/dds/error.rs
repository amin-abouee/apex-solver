//! Errors of the live DDS subscriber.

/// Why a DDS subscription could not be set up.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DdsError {
    #[error("DDS participant creation failed on domain {domain_id}")]
    ParticipantCreation {
        domain_id: u16,
        #[source]
        source: rustdds::dds::CreateError,
    },

    #[error("DDS topic creation failed for {topic}")]
    TopicCreation {
        topic: String,
        #[source]
        source: rustdds::dds::CreateError,
    },

    #[error("DDS subscriber creation failed")]
    SubscriberCreation(#[source] rustdds::dds::CreateError),

    #[error("DDS DataReader creation failed for {topic}")]
    DataReaderCreation {
        topic: String,
        #[source]
        source: rustdds::dds::CreateError,
    },

    #[error("invalid topic name {name:?}: {reason}")]
    InvalidTopicName { name: String, reason: &'static str },

    /// Failed to spawn the OS thread that drives the DDS reader loop.
    #[error("failed to spawn the DDS reader thread for {topic}")]
    ThreadSpawn {
        topic: String,
        #[source]
        source: std::io::Error,
    },

    /// Failed to construct the tokio runtime the reader loop runs on.
    #[error("failed to create the tokio runtime for the DDS reader")]
    RuntimeCreation(#[source] std::io::Error),
}

pub type Result<T> = std::result::Result<T, DdsError>;

/// Result of a DDS operation, named like the other module aliases.
pub type DdsResult<T> = Result<T>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorChainExt;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn dds_error_is_send_sync() {
        assert_send_sync::<DdsError>();
    }

    #[test]
    fn invalid_topic_name_display_includes_name_and_reason() {
        let err = DdsError::InvalidTopicName {
            name: "bad_topic".to_string(),
            reason: "must not be empty",
        };
        assert_eq!(
            err.to_string(),
            "invalid topic name \"bad_topic\": must not be empty"
        );
    }

    #[test]
    fn thread_spawn_keeps_its_source() {
        let err = DdsError::ThreadSpawn {
            topic: "/imu".to_string(),
            source: std::io::Error::other("resource exhaustion"),
        };
        assert_eq!(
            err.chain_compact(),
            "failed to spawn the DDS reader thread for /imu → resource exhaustion"
        );
    }

    #[test]
    fn runtime_creation_keeps_its_source() {
        let err = DdsError::RuntimeCreation(std::io::Error::other("no tokio driver"));
        assert_eq!(
            err.chain_compact(),
            "failed to create the tokio runtime for the DDS reader → no tokio driver"
        );
    }
}
