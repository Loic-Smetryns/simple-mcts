use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use simple_mcts::{negate_score, puct, Action, Engine, MctsBatch, MctsConfig, MctsStateEvaluation, StateEvaluation};

fn run_simulations(mcts: &mut Engine::<9>, n: usize) {
    const C: f32 = 1.414;
    let mut path = Vec::with_capacity(16);

    for _ in 0..n {
        let selection = mcts.select(&mut path, &puct, C);
        let eval = StateEvaluation::Active(0.0, [1./9.; 9]);
        mcts.update(eval, selection, negate_score).unwrap();
    }

    black_box(mcts);
}

fn loops_with_engine(c: &mut Criterion) {
    let mut group = c.benchmark_group("loops_with_engine");
    group.measurement_time(std::time::Duration::from_secs(25));
    group.sample_size(100);

    group.bench_function("1K updates", |b| {
        b.iter_batched(
            || Engine::<9>::new(),
            |mut mcts| run_simulations(&mut mcts, 1_000),
            BatchSize::SmallInput
        )
    });

    group.bench_function("5K updates", |b| {
        b.iter_batched(
            || Engine::<9>::new(),
            |mut mcts| run_simulations(&mut mcts, 5_000),
            BatchSize::SmallInput
        )
    });

    group.sample_size(50);

    group.bench_function("10k updates", |b| {
        b.iter_batched(
            || Engine::<9>::new(),
            |mut mcts| run_simulations(&mut mcts, 10_000),
            BatchSize::SmallInput
        )
    });

    group.bench_function("50k updates", |b| {
        b.iter_batched(
            || Engine::<9>::new(),
            |mut mcts| run_simulations(&mut mcts, 50_000),
            BatchSize::SmallInput
        )
    });

    group.sample_size(20);

    group.bench_function("100k updates", |b| {
        b.iter_batched(
            || Engine::<9>::new(),
            |mut mcts| run_simulations(&mut mcts, 100_000),
            BatchSize::SmallInput
        )
    });

    group.sample_size(10);

    group.bench_function("1M updates", |b| {
        b.iter_batched(
            || Engine::<9>::new(),
            |mut mcts| run_simulations(&mut mcts, 1_000_000),
            BatchSize::SmallInput
        )
    });

    group.finish();
}

fn init_mcts(n: usize) -> Engine::<9> {
    let mut mcts = Engine::<9>::new();

    const C: f32 = 1.414;
    let mut path = Vec::with_capacity(16);

    for _ in 0..n {
        let selection = mcts.select(&mut path, &puct, C);
        let eval = StateEvaluation::Active(0.0, [1./9.; 9]);
        mcts.update(eval, selection, negate_score).unwrap();
    }

    mcts
}
fn run_compact(mcts: &mut Engine::<9>){
    mcts.commit_action(Action::new(0));
    black_box(mcts);
}

fn compact(c: &mut Criterion) {
    let mut group = c.benchmark_group("commit_action");
    group.measurement_time(std::time::Duration::from_secs(25));
    group.sample_size(100);

    group.bench_function("compact - 1K", |b| {
        b.iter_batched(
            || init_mcts(1_000),
            |mut mcts| run_compact(&mut mcts),
            BatchSize::LargeInput
        )
    });

    group.bench_function("compact - 5K", |b| {
        b.iter_batched(
            || init_mcts(5_000),
            |mut mcts| run_compact(&mut mcts),
            BatchSize::LargeInput
        )
    });

    group.sample_size(50);

    group.bench_function("compact - 10K", |b| {
        b.iter_batched(
            || init_mcts(10_000),
            |mut mcts| run_compact(&mut mcts),
            BatchSize::LargeInput
        )
    });

    group.bench_function("compact - 50K", |b| {
        b.iter_batched(
            || init_mcts(50_000),
            |mut mcts| run_compact(&mut mcts),
            BatchSize::LargeInput
        )
    });

    group.sample_size(20);

    group.bench_function("compact - 100K", |b| {
        b.iter_batched(
            || init_mcts(100_000),
            |mut mcts| run_compact(&mut mcts),
            BatchSize::LargeInput
        )
    });

    group.sample_size(10);

    group.bench_function("compact - 1M", |b| {
        b.iter_batched(
            || init_mcts(1_000_000),
            |mut mcts| run_compact(&mut mcts),
            BatchSize::LargeInput
        )
    });
}

fn run_batch(n: usize){
    let config = MctsConfig::new(std::f32::consts::SQRT_2, puct, negate_score);
    let mut mcts_batch = MctsBatch::<_, _, 9>::from_config(config);

    mcts_batch.populate(n);

    for _ in 0..10{
        let selection = mcts_batch.selection();

        let mut evaluations = selection.into_iter().map(|selection| MctsStateEvaluation {
            id: selection.id,
            selection: selection.selection,
            evaluation: StateEvaluation::Active(0.0, [1./9.; 9])
        }).collect();

        mcts_batch.update(&mut evaluations).unwrap();
    }

    black_box(mcts_batch);
}

fn batch(c: &mut Criterion){
    let mut group = c.benchmark_group("batch");
    group.measurement_time(std::time::Duration::from_secs(25));
    group.sample_size(10);

    group.bench_function("batch - 1K", |b| {
        b.iter(
            || run_batch(1_000),
        )
    });


    group.bench_function("batch - 5K", |b| {
        b.iter(
            || run_batch(5_000),
        )
    });


    group.bench_function("batch - 10K", |b| {
        b.iter(
            || run_batch(10_000),
        )
    });


    group.bench_function("batch - 50K", |b| {
        b.iter(
            || run_batch(50_000),
        )
    });


    group.bench_function("batch - 100K", |b| {
        b.iter(
            || run_batch(100_000),
        )
    });
}

fn bench(c: &mut Criterion) {
    loops_with_engine(c);
    compact(c);
    batch(c);
}

criterion_group!(benches, bench);
criterion_main!(benches);