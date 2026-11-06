# How I got to this

This is the longer story behind caddy. The [README](../README.md) has the short version, and [Notes](../Notes.md) has
the decisions, limitations and next steps.

## Starting with the market

I didn't start with code. I already run agents that do a daily literature review. They read company blogs, GitHub,
LinkedIn and YouTube and compare what they find with my work. Going through that output
pointed me at areas worth a closer look, but it didn't give me an answer. The clearest signal came from talking to
companies. They keep asking for the same thing: take their CAD drawings and turn them into context a model can use.

## What users of the closest tool complain about

Next I read the open issues on Adam's open-source [CADAM](https://github.com/Adam-CAD/CADAM), plus Reddit and the
[launch thread](https://news.ycombinator.com/item?id=47977694). I was looking for what people say when something
annoys them, and three things kept coming up.

- Simple edits take too long and burn tokens. I've heard the same from engineers at work.
- Engineers want "full visibility and control over the feature tree," in the founder's words on the launch thread.
  When people use the copilot, they ask it to "clean up my tree" or "parametrize my model." Those are requests to
  understand and edit a model, not to generate one.
- People can't pick out one component and work with it ([#62](https://github.com/Adam-CAD/CADAM/issues/62)). One
  user stopped using the tool because there was no manual editing
  ([#79](https://github.com/Adam-CAD/CADAM/issues/79)), and someone proposed a design tree
  ([#139](https://github.com/Adam-CAD/CADAM/issues/139)).

## Picking a problem

I had four candidates:

- sketch-to-CAD
- verification in the loop
- McMaster parts
- a benchmark-driven test set in the style of BenchCAD

I went with verification in the loop. The easiest way to describe it is Simulink for CAD: the model gets built, and
then it's checked against requirements you can read.

Sketch-to-CAD is mostly a generation problem, and the complaints I'd found were about editing and cost. McMaster files
come with licensing questions. So the parts come from a public catalogue instead: a reducing hex nipple datasheet from
NERO, plus the ISO 7-1 thread standard. The sources are in [`data/SOURCES.md`](../data/SOURCES.md).

This is where the thesis took shape. AI CAD is expensive for simple things because the agent talks to the CAD tool
through pictures and full rewrites. NVIDIA's B-rep kernel can sit between them. It turns exact geometry into a
compact, named description, and edits go back as small diffs against that description. Text works better than pixels
here: BenchCAD reports about 0.84 when a model answers questions about a part from its code, and about 0.59 from
images.

The second half came from the engineers. Once a model exists, people want to change it, and you can't change what you
don't understand. So the same named description has to work for two readers, the model and the engineer.

## What I brought from earlier projects

Three of my earlier projects fit:

- **Whiteboard** is a live, visual agent you can watch and steer. Here it became the board that shows each step.
- **[Local design](https://github.com/ghanemja/localdesign)** is a Claude Design-style tool built around a design system, the tokens and rules that every screen
  follows. A company slide template is a design system. A company CAD standard is the same idea for geometry:
  preferred hole sizes, minimum wall thickness, fillet sizes and tolerance classes. The kernel can check a solid
  against those rules the way a design check looks at a screen.
- **Behaviors** brought evaluation practice. Later it can bring my JEPA model for behaviour search, so people can look
  for parts by how they perform and not just by what kind of object they are.

The team CAD standard, the context graph and behaviour search are part of the plan, but they aren't built yet. They're
listed as vision in [Tasks](TASKS.md).

## Why this kernel

I used NVIDIA SMLib as the only kernel ([ADR-1](DECISIONS.md)). It gives exact B-rep geometry, it's open, and it
writes USD. It also has costs: there's no STEP writer, and the wheel only exists for Linux and Windows, which is why
the project runs in Docker on a Mac.

The kernel isn't always right, either. Its distance query read a 40 mm gap between a tyre and an arch as touching, on
lofted surfaces. Now, when it reports zero on a skinned body that shares no volume with its neighbour, the code
measures again on the exact surfaces ([ADR-7](DECISIONS.md)).

## Planning

Before the build, I wrote down what had to be true at each stage and what I'd cut if it wasn't. What actually
shipped is in [Milestones](MILESTONES.md) and the [Changelog](CHANGELOG.md).

## The first real surprise

The first build found a problem I hadn't planned for. On both ¾-inch rows, the large end of the inlet thread is wider
than the hex across-flats, so the thread cone breaks through the end face of the hex. That gave 38 faces where there
should be 27, plus slivers.

The catalogue gives minimum dimensions, and real parts have a thread runout. So the rule became "the thread must
clear the hex," and a face-count check caught the failure. That's the failure the agent learned to repair. Details
are in [Notes](../Notes.md) (assumptions).

## What changed once the core worked

Once the nipple loop worked end to end, I widened it on purpose, to test whether the idea holds beyond one family:

- an assembly (Adam's SO-101 gripper demo task);
- scan → CAD for turned parts;
- freeform bodies and wings on the kernel (a Roadster-style coupe, then a flying car);
- drawings and an aero estimate built from the same model.

The flying car came from a post on X by Zach: "the flying cars of the future will be designed by agents." That
inspired the demo case. Claude starts from the coupe and adds wings, lift fans and a tail, and the kernel
checks every step.

The original plan left assemblies and Docker out, so both changes are recorded with their reasons under "Decisions"
in [Notes](../Notes.md).

## Measuring instead of trusting

The model doesn't get to declare a build done: `finish` is refused while any check fails ([ADR-4](DECISIONS.md)).
Fixes go in as small text diffs, and slider changes go straight to the kernel with no model call
([ADR-5](DECISIONS.md)). [VERIFY.md](VERIFY.md) maps each requirement to the test that proves it, and lists the
requirements that aren't covered yet. The one comparison I haven't measured is caddy against today's
screenshot-and-rewrite loop on the same task. That's the first item in [Tasks](TASKS.md).
