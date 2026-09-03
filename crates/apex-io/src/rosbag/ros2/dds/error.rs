use thiserror::Error;

#[derive(Error, Debug)]
pub enum DdsError {
    #[error("DDS participant creation failed: {0}")]
    ParticipantCreation(String),

    #[error("DDS topic creation failed for '{topic}': {reason}")]
    TopicCreation { topic: String, reason: String },

    #[error("DDS subscriber creation failed: {0}")]
    SubscriberCreation(String),

    #[error("DDS DataReader creation failed: {0}")]
    DataReaderCreation(String),

    #[error("DDS receive error: {0}")]
    Receive(String),

    #[error("Channel send error: message receiver was dropped")]
    ChannelSend,

    #[error("Invalid topic name '{name}': {reason}")]
    InvalidTopicName { name: String, reason: String },

    #[error("Thread join failed: {0}")]
    ThreadJoin(String),

    /// Failed to spawn the OS thread that drives the DDS reader loop.
    #[error("Failed to spawn DDS reader thread: {0}")]
    ThreadSpawn(String),

    /// Failed to construct the tokio runtime the reader loop runs on.
    #[error("Failed to create tokio runtime for DDS reader: {0}")]
    RuntimeCreation(String),
}

pub type Result<T> = std::result::Result<T, DdsError>;

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn dds_error_is_send_sync() {
        assert_send_sync::<DdsError>();
    }

    #[test]
    fn invalid_topic_name_display_includes_name_and_reason() {
        let err = DdsError::InvalidTopicName {
            name: "bad_topic".to_string(),
            reason: "must not be empty".to_string(),
        };
        let s = err.to_string();
        assert!(s.contains("bad_topic"));
        assert!(s.contains("must not be empty"));
    }

    #[test]
    fn participant_creation_display() {
        let err = DdsError::ParticipantCreation("network error".to_string());
        assert!(err.to_string().contains("network error"));
    }

    #[test]
    fn topic_creation_display_includes_topic() {
        let err = DdsError::TopicCreation {
            topic: "/cam".to_string(),
            reason: "duplicate".to_string(),
        };
        let s = err.to_string();
        assert!(s.contains("/cam"));
        assert!(s.contains("duplicate"));
    }

    #[test]
    fn channel_send_display() {
        let err = DdsError::ChannelSend;
        assert!(err.to_string().contains("receiver"));
    }

    #[test]
    fn thread_join_display() {
        let err = DdsError::ThreadJoin("OS error".to_string());
        assert!(err.to_string().contains("OS error"));
    }

    #[test]
    fn thread_spawn_display() {
        let err = DdsError::ThreadSpawn("resource exhaustion".to_string());
        let s = err.to_string();
        assert!(s.contains("spawn"));
        assert!(s.contains("resource exhaustion"));
    }

    #[test]
    fn runtime_creation_display() {
        let err = DdsError::RuntimeCreation("no tokio driver".to_string());
        let s = err.to_string();
        assert!(s.contains("runtime"));
        assert!(s.contains("no tokio driver"));
    }
}
