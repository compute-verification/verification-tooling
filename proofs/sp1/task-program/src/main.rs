#![no_main]

use pocomp_protocol::{
    commitment, evaluate_task_relation, RelationPublicValues, TaskPublicStatement,
    TaskRelationInput,
};

sp1_zkvm::entrypoint!(main);

pub fn main() {
    let input = sp1_zkvm::io::read::<TaskRelationInput>();
    let outcome = evaluate_task_relation(&input).expect("Task-PoComp relation failed");
    let statement = TaskPublicStatement::from(&input);
    sp1_zkvm::io::commit(&RelationPublicValues {
        statement_digest: commitment(&statement),
        outcome,
    });
}
