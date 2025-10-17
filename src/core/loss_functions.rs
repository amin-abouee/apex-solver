use core::f64;

pub trait Loss: Send + Sync {
    fn evaluate(&self, s: f64) -> [f64; 3];
}

#[derive(Debug, Clone)]
pub struct HuberLoss {
    scale: f64,
    scale2: f64,
}

impl HuberLoss {
    pub fn new(scale: f64) -> crate::core::ApexResult<Self> {
        if scale <= 0.0 {
            return Err(crate::core::ApexError::InvalidInput(
                "scale needs to be larger than zero".to_string(),
            ));
        }
        Ok(HuberLoss {
            scale,
            scale2: scale * scale,
        })
    }
}

impl Loss for HuberLoss {
    fn evaluate(&self, s: f64) -> [f64; 3] {
        if s > self.scale2 {
            // Outlier region.
            // 'r' is always positive.
            let r = s.sqrt();
            let rho1 = (self.scale / r).max(f64::MIN);
            [2.0 * self.scale * r - self.scale2, rho1, -rho1 / (2.0 * s)]
        } else {
            // Inlier region.
            [s, 1.0, 0.0]
        }
    }
}

pub struct CauchyLoss {
    scale2: f64,
    c: f64,
}

impl CauchyLoss {
    pub fn new(scale: f64) -> crate::core::ApexResult<Self> {
        if scale <= 0.0 {
            return Err(crate::core::ApexError::InvalidInput(
                "scale needs to be larger than zero".to_string(),
            ));
        }
        let scale2 = scale * scale;
        Ok(CauchyLoss {
            scale2,
            c: 1.0 / scale2,
        })
    }
}

impl Loss for CauchyLoss {
    fn evaluate(&self, s: f64) -> [f64; 3] {
        let sum = 1.0 + s * self.c;
        let inv = 1.0 / sum;
        // 'sum' and 'inv' are always positive, assuming that 's' is.
        [
            self.scale2 * sum.log2(),
            inv.max(f64::MIN),
            -self.c * (inv * inv),
        ]
    }
}
