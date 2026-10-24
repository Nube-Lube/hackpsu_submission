mod gemini;

// Step 1: `cargo run -- photo.jpg` sends one image to Gemini and prints the hazards.
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let path = std::env::args().nth(1).expect("usage: hackpsu <image.jpg>");
    let jpeg = std::fs::read(&path)?;

    let report = gemini::Gemini::from_env()?.analyze(&jpeg).await?;
    println!("{report:#?}");
    Ok(())
}
