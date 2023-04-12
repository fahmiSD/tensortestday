let x_vals = [];
let y_vals = [];

let a, b, c, d, e;
let dragging = false;


const learningRate = 0.5;
const optimazir = tf.train.sgd(learningRate);

function setup() {
    createCanvas(400, 400);

    a = tf.variable(tf.scalar(random(-1, 1)));
    b = tf.variable(tf.scalar(random(-1, 1)));
    c = tf.variable(tf.scalar(random(-1, 1)));
    d = tf.variable(tf.scalar(random(-1, 1)));
    e = tf.variable(tf.scalar(random(-1, 1)));
}

function loss(pred, labels) {
    return pred.sub(labels).square().mean();
}

function predict(x) {

    const xs = tf.tensor1d(x);
    // y = mx + b (linear regression)
    // const ys = xs.mul(m).add(b);

    // y = ax^2+bx+c (polinomial regression)
    // const ys = xs.square().mul(a).add(xs.mul(b)).add(c);

    // y = ax ^ 3 + bx ^ 2 + cx + d
    // const ys = xs.pow(tf.scalar(3)).mul(a)
    //     .add(xs.square().mul(b))
    //     .add(xs.mul(c))
    //     .add(d);

    // y = a + b*(xs/305) + c*(xs/305)^2 + d*ln(305/xs) + e*(ln(305/xs)^2)
    const ys = a
        .add((xs.div(tf.scalar(305))).mul(b))
        .add((xs.div(tf.scalar(305))).square().mul(c))
        .add(((tf.scalar(305).div(xs)).log()).mul(d))
        .add(((tf.scalar(305).div(xs)).log().square()).mul(e));
    return ys;
}

function mousePressed() {
    dragging = true;
}

function mouseReleased() {
    dragging = false;
}

// function mouseDragged() {
//     let x = map(mouseX, 0, width, -1, 1);
//     let y = map(mouseY, 0, height, 1, -1);

//     x_vals.push(x);
//     y_vals.push(y);
// }

function draw() {

    if (dragging) {
        let x = map(mouseX, 0, width, -1, 1);
        let y = map(mouseY, 0, height, 1, -1);

        x_vals.push(x);
        y_vals.push(y);
    }
    if (!dragging) {
        tf.tidy(() => {

            if (x_vals.length > 0) {

                const ys = tf.tensor1d(y_vals)
                optimazir.minimize(() => loss(predict(x_vals), ys));

            }

        });
    }
    // console.log(dragging);

    background(0);
    stroke(255);
    strokeWeight(6);
    for (let i = 0; i < x_vals.length; i++) {
        let px = map(x_vals[i], -1, 1, 0, width);
        let py = map(y_vals[i], -1, 1, height, 0);
        point(px, py);
    }

    tf.tidy(() => {
        const curveX = [];
        for (let x = -1; x < 1.01; x += 0.05) {
            curveX.push(x);
        }

        // console.log(curveX);
        const ys = predict(curveX);

        let curveY = ys.dataSync();

        beginShape();
        noFill();
        stroke(255);
        strokeWeight(2);

        for (let i = 0; i < curveX.length; i++) {
            let x = map(curveX[i], -1, 1, 0, width);
            let y = map(curveY[i], -1, 1, height, 0);
            vertex(x, y);
        }
        endShape();

        console.log(tf.memory().numTensors);
    });

}

