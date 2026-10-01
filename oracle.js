/**
 * The oracle: a small, fixed set of reflex rules, not a model.
 *
 * Given the state returned by window.__getState() in game.html, decides one
 * of "left" | "right" | "jump" | "duck" | "none". This is the "teacher"
 * policy -- play_oracle.js runs it and logs every (state, action) pair as a
 * training example for fine-tuning Laya to imitate it.
 */

const REACT_DISTANCE = 15;

export function decideAction(state) {
  const blocking = state.obstaclesAhead.find(
    (o) => o.lane === state.currentLane && o.distance > 0 && o.distance < REACT_DISTANCE
  );

  if (!blocking) return "none";

  if (blocking.type === "train") {
    // Can't jump or duck a train -- must change lanes entirely.
    return state.currentLane === 0 ? "right" : "left";
  }
  if (blocking.isHigh) {
    return "duck";
  }
  return "jump";
}
