//! Core optimization components for the apex-solver library
//!
//! This module contains the fundamental building blocks for nonlinear least squares optimization:
//! - Problem formulation and management
//! - Residual blocks and factors
//! - Variables and manifold handling
//! - Loss functions for robust estimation
//! - Correctors for applying loss functions

pub mod corrector;
pub mod factors;
pub mod loss_functions;
pub mod problem;
pub mod residual_block;
pub mod variable;
