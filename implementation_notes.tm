<TeXmacs|1.99.13>

<style|generic>

<\body>
  <doc-data|<doc-title|Soft actor-critic>>

  <section|The SAC algorithm>

  <subsection|How to calculate <math|log \<pi\><rsub|\<theta\>><around|(|<wide|a|~><rsub|\<theta\>><around|(|s|)>\<mid\>s|)>>
  properly?>

  In Step 14, we have

  <\equation*>
    \<nabla\><rsub|\<theta\>> <frac|1|<around|\||B|\|>>
    <big|sum><rsub|s\<in\>B><around|<left|(|3>|min<rsub|i=1,2>Q<rsub|\<phi\><rsub|i>><around|(|s,<wide|a|~><rsub|\<theta\>><around|(|s|)>|)>-\<alpha\>
    log \<pi\><rsub|\<theta\>><around|(|<wide|a|~><rsub|\<theta\>><around|(|s|)>\<mid\>s|)>|<right|)|3>>,
  </equation*>

  where in the second term both <math|\<pi\>> and <math|<wide|a|~>> depends
  on <math|\<theta\>>. This was confusing to me. How is backprop possible
  here? If it is possible, is it appropriate?

  To start off, note from the paper that

  <\equation*>
    log \<pi\><rsub|\<theta\>><around|(|a\<mid\>s|)>=log
    \<mu\><rsub|\<theta\>><around|(|u\<mid\>s|)>-log<around|(|1-tanh<rsup|2><around|(|u|)>|)>
  </equation*>

  where <math|u\<sim\>\<mu\><around|(|means<rsub|\<theta\>><around|(|s|)>,stds<rsub|\<theta\>><around|(|s|)>|)>>
  and <math|\<mu\>> denotes a diagonal normal distribution. Note crucially
  that <math|u> is sampled through reparametrization, i.e.,
  <math|u=means<rsub|\<theta\>><around|(|s|)>+\<varepsilon\>\<cdot\>stds<rsub|\<theta\>><around|(|s|)>>
  where <math|\<varepsilon\>\<sim\><with|font|cal|N><around|(|0,1|)>>. So, my
  confusion was not about backpropagating through <math|log
  \<pi\><rsub|\<theta\>><around|(|<wide|a|~><rsub|\<theta\>><around|(|s|)>\<mid\>s|)>>.
  Instead, it was about backpropagating through <math|log
  \<mu\><rsub|\<theta\>><around|(|u\<mid\>s|)>>.

  Acknowledging the true source of my confusion, I turned attention to
  <with|font-series|bold|a simpler problem>. Just as before (except that we
  are being one-dimensional and ignoring the dependency of mean and std on
  additional parameters), suppose <math|u\<sim\>\<mu\><around|(|mean,std|)>>
  and <math|u=mean+\<varepsilon\>\<cdot\>std> where
  <math|\<varepsilon\>\<sim\><with|font|cal|N><around|(|0,1|)>>.

  Just as before, we are interested in the quantity

  <\eqnarray*>
    <tformat|<table|<row|<cell|-log \<mu\><around|(|u|)>>|<cell|=>|<cell|<frac|1|std<sqrt|2\<pi\>>>
    exp<around|<left|[|3>|-<frac|1|2><around|<left|(|2>|<frac|u-mean|std>|<right|)|2>><rsup|2>|<right|]|3>><space|1em><around|(|<text|good
    old normal>|)>>>|<row|<cell|>|<cell|=>|<cell|<frac|1|std<sqrt|2\<pi\>>>
    exp<around|<left|[|3>|-<frac|1|2><around|<left|(|2>|<frac|mean+\<varepsilon\>\<cdot\>std-mean|std>|<right|)|2>><rsup|2>|<right|]|3>><space|1em>>>|<row|<cell|>|<cell|=>|<cell|<frac|1|std<sqrt|2\<pi\>>>
    exp<around|<left|[|3>|-<frac|1|2><around|<left|(|2>|<frac|\<varepsilon\>\<cdot\>std|std>|<right|)|2>><rsup|2>|<right|]|3>>>>|<row|<cell|>|<cell|=>|<cell|<frac|1|std<sqrt|2\<pi\>>>
    exp<around|<left|[|3>|-<frac|1|2><around|<left|(|2>|\<varepsilon\>|<right|)|2>><rsup|2>|<right|]|3>>>>>>
  </eqnarray*>

  which is a little surprising because the gradient with respect to mean will
  be zero. However, this is intuitively <with|font-shape|italic|correct> in
  that increasing entropy corresponds to increasing the spread of the
  distribution and not the location of its center.\ 

  Of course, in PyTorch, no explicit cancellation of mean is not carried;
  rather, it is calculated in run time as long as we sample in a
  differentiable way (i.e., using the reparameterization trick) with the
  rsample function. Below is an example where <math|u>, means and stds are
  5-dimensional. As you can see, the gradients with respect of means are
  zeros. Try using sample instead of rsample; then you will see that the
  gradients are not zeros.

  <\python-code>
    import torch

    from torch.distributions import Normal, Independent

    \;

    means = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
    dtype=torch.float).view(2, -1)

    stds \ = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
    dtype=torch.float).view(2, -1)

    \;

    means.requires_grad = True

    stds.requires_grad = True

    \;

    dist = Independent(Normal(loc=means, scale=stds), 1)

    \;

    print(dist.batch_shape, dist.event_shape) \ # torch.Size([2])
    torch.Size([5])

    \;

    samples_with_grad = dist.rsample(sample_shape=torch.Size([]))

    \;

    print(samples_with_grad.shape) \ # torch.Size([2, 5])

    \;

    log_prob = - dist.log_prob(samples_with_grad)

    result = torch.sum(log_prob)

    result.backward()

    \;

    print(means.grad)

    print(stds.grad)

    \;

    # tensor([[0., 0., 0., 0., 0.],

    # \ \ \ \ \ \ \ \ [0., 0., 0., 0., 0.]])

    # tensor([[1.0000, 0.5000, 0.3333, 0.2500, 0.2000],

    # \ \ \ \ \ \ \ \ [1.0000, 0.5000, 0.3333, 0.2500, 0.2000]])
  </python-code>

  Now, <with|font-series|bold|going back to the original problem>, there are
  two more things to talk about. First, the means and stds now depend on
  parameters; this is easy to understand so I won't talk about this. Second,
  we are ultimately interested in understanding the properties of gradients
  of <math|log \<pi\><rsub|\<theta\>><around|(|a\<mid\>s|)>> instead of
  <math|log \<mu\><rsub|\<theta\>><around|(|u\<mid\>s|)>>.\ 

  <\eqnarray*>
    <tformat|<table|<row|<cell|log \<pi\><rsub|\<theta\>><around|(|a\<mid\>s|)>>|<cell|=>|<cell|log
    \<mu\><rsub|\<theta\>><around|(|u\<mid\>s|)>-log<around|(|1-tanh<rsup|2><around|(|u|)>|)>>>|<row|<cell|<frac|d|d
    means> log \<pi\><rsub|\<theta\>><around|(|a\<mid\>s|)>>|<cell|=>|<cell|0-<around|(|<text|some
    non-zero term>|)>>>>>
  </eqnarray*>

  Having a non-zero gradient in this case is intuitively correct because,
  since <math|\<pi\>> is actually a normal distribution squashed by tanh, it
  has to move it means closer to zero to increase it's entropy.

  \;

  \;

  \;

  \;

  \;

  \;
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
    <associate|page-screen-margin|false>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1|../../../.TeXmacs/texts/scratch/no_name_6.tm>>
    <associate|auto-2|<tuple|1.1|1|../../../.TeXmacs/texts/scratch/no_name_6.tm>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>The
      SAC algorithm> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <with|par-left|<quote|1tab>|1.1<space|2spc>How to calculate
      <with|mode|<quote|math>|log \<pi\><rsub|\<theta\>><around|(|<wide|a|~><rsub|\<theta\>><around|(|s|)>\<mid\>s|)>>
      properly? <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>
    </associate>
  </collection>
</auxiliary>