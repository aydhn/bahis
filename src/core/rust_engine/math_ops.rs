/*
 math_ops.rs – Quant Betting Bot için yüksek performanslı matematik çekirdeği.
*/

use std::f64::consts::PI;

#[no_mangle]
pub extern "C" fn calculate_kelly_rust(edge: f64, odds: f64) -> f64 {
    if odds <= 1.0 { return 0.0; }
    let f = (edge * odds - (1.0 - edge)) / odds;
    if f < 0.0 { 0.0 } else { f }
}

#[no_mangle]
pub extern "C" fn fast_monte_carlo(n_sim: i32, prob: f64) -> f64 {
    // Rust üzerinde 1M+ simülasyonu milisaniyeler içinde yap
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut wins = 0;
    
    for _ in 0..n_sim {
        if rng.gen_bool(prob) {
            wins += 1;
        }
    }
    (wins as f64) / (n_sim as f64)
}

#[no_mangle]
pub extern "C" fn gaussian_copula_sample(u: f64, v: f64, rho: f64) -> f64 {
    // İki varyant arasındaki bağımlılığı (Copula) hesapla
    // Basit bivariate normal approximation
    let term1 = - (u.powi(2) - 2.0 * rho * u * v + v.powi(2)) / (2.0 * (1.0 - rho.powi(2)));
    (1.0 / (2.0 * PI * (1.0 - rho.powi(2)).sqrt())) * term1.exp()
}
