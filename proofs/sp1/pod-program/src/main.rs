#![no_main]

use pocomp_protocol::{
    commitment, evaluate_pod_relation, PodPublicStatement, PodRelationInput, RelationPublicValues,
};

sp1_zkvm::entrypoint!(main);

pub fn main() {
    let input = sp1_zkvm::io::read::<PodRelationInput>();
    let outcome = evaluate_pod_relation(&input).expect("Pod-PoComp relation failed");
    let statement = PodPublicStatement::from(&input);
    sp1_zkvm::io::commit(&RelationPublicValues {
        statement_digest: commitment(&statement),
        outcome,
    });
}
