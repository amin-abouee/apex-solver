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
}

pub type Result<T> = std::result::Result<T, DdsError>;
