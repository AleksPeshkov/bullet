use bulletformat::ChessBoard;

use super::SparseInputType;

#[derive(Clone, Copy, Debug, Default)]
pub struct Chess768hm;
impl SparseInputType for Chess768hm {
    type RequiredDataType = ChessBoard;

    /// The total number of inputs
    fn num_inputs(&self) -> usize {
        768
    }

    /// The maximum number of active inputs
    fn max_active(&self) -> usize {
        32
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let hm_s = if (pos.our_ksq() & 4) != 0 { 0 } else { 7 };
        let hm_n = if (pos.opp_ksq() & 4) != 0 { 0 } else { 7 };

        for (piece, square) in pos.into_iter() {
            let c = usize::from(piece & 8 > 0);
            let pc = 64 * usize::from(piece & 7);
            let sq = usize::from(square);

            let stm = [0, 384][c] + pc + (sq ^ hm_s);
            let ntm = [384, 0][c] + pc + (sq ^ hm_n ^ 56);
            f(stm, ntm)
        }
    }

    /// Shorthand for the input e.g. `768x4`
    fn shorthand(&self) -> String {
        "768hm".to_string()
    }

    /// Description of the input type
    fn description(&self) -> String {
        "Psqt chess inputs with hm".to_string()
    }
}
