use nn_core::*;

fn acc(net: &Mlp, data: &[(Vec<f64>, f64)]) -> f64 {
    let ok = data.iter().filter(|(x, y)| ((net.predict(x)[0] > 0.5) as i32 as f64) == *y).count();
    ok as f64 / data.len() as f64
}

fn main() {
    let data = Dataset::Spiral.points(7, 100);
    let hiddens = vec![4, 8, 16];
    let lrs = vec![0.1, 0.5, 1.0];
    
    println!("hidden,lr,acc_10k,acc_20k,acc_40k,final_loss");
    
    for &hidden in &hiddens {
        for &lr in &lrs {
            let mut rng = Rng::new(42);
            let mut net = Mlp::new(&[2, hidden, 1], &mut rng);
            
            let mut acc_10k = 0.0;
            let mut acc_20k = 0.0;
            let mut final_loss = 0.0;
            
            for step in 1..=40000 {
                final_loss = net.sgd_step(&data, lr);
                
                if step == 10000 {
                    acc_10k = acc(&net, &data);
                } else if step == 20000 {
                    acc_20k = acc(&net, &data);
                }
            }
            
            let acc_40k = acc(&net, &data);
            
            println!("{},{},{:.4},{:.4},{:.4},{:.6}", hidden, lr, acc_10k, acc_20k, acc_40k, final_loss);
        }
    }
}
