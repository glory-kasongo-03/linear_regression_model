use burn::tensor::Tensor;
use burn_ndarray::NdArrayBackend;
use rand::Rng;
use textplots::{Chart, Plot, Shape};
use rgb::RGB; // Import the RGB struct from the rgb crate

fn main() {
    // Generate synthetic data
    let mut rng = rand::rng();
    let n_samples = 100;
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    for _ in 0..n_samples {
        let x: f32 = rng.random_range(0.0..10.0);
        let noise: f32 = rng.random_range(-1.0..1.0);
        let y = 2.0 * x + 1.0 + noise;
        x_data.push(x);
        y_data.push(y);
    }

    // Convert data to tensors
    let x_tensor = Tensor::<NdArrayBackend<f32>, 1>::from_data(x_data.clone(), &());
    let y_tensor = Tensor::<NdArrayBackend<f32>, 1>::from_data(y_data.clone(), &());

    // Plot the synthetic data
    Chart::new(180, 60, 0.0, 10.0)
        .lineplot(&Shape::Points(&x_data.iter().zip(y_data.iter()).map(|(&x, &y)| (x, y)).collect::<Vec<_>>()))
        .display();

    // Linear regression model (adjusting to ensure types are correct)
    let mut w = Tensor::<NdArrayBackend<f32>, 1>::from_data(vec![0.0], &());
    let mut b = Tensor::<NdArrayBackend<f32>, 1>::from_data(vec![0.0], &());

    // Training loop
    let learning_rate = 0.01;
    let epochs = 1000;

    for epoch in 0..epochs {
        // Clone tensors to avoid borrowing immutably while mutating w and b
        let y_pred = &x_tensor.clone() * &w.clone() + &b.clone();
        let loss = (&y_pred - &y_tensor).pow(2).mean();

        let grad_w = 2.0 * (&x_tensor * (&y_pred - &y_tensor)).mean();
        let grad_b = 2.0 * (&y_pred - &y_tensor).mean();

        // In-place update of w and b (passing both parameters)
        w.set(w.clone(), w.clone() - grad_w * learning_rate);
        b.set(b.clone(), b.clone() - grad_b * learning_rate);

        // Example: Use rgb to show a color change in each epoch
        // Change the RGB values based on the current epoch or loss
        let color = RGB::new(
            (epoch % 256) as u8,    // Red value based on epoch
            (loss * 100.0) as u8,   // Green value based on loss
            255 - (epoch % 256) as u8, // Blue value based on epoch
        );

        // Log the color at every 100th epoch (just an example)
        if epoch % 100 == 0 {
            println!("Epoch {}: Color: ({}, {}, {})", epoch, color.r, color.g, color.b);
        }
    }

    let w_data = w.data();
    let b_data = b.data();

    // Accessing data directly
    println!("Trained model: y = {:.2}x + {:.2}", w_data[0], b_data[0]);
}
